"""
Image Processor for Multi-Modal RAG
Handles image loading, preprocessing, and metadata extraction.
"""

import io
import base64
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Processed image with metadata."""
    image_id: str
    width: int
    height: int
    format: str
    file_size: int
    path: Optional[str]
    base64_thumbnail: Optional[str]
    extracted_text: Optional[str]
    detected_objects: List[str]
    metadata: Dict[str, Any]


class ImageProcessor:
    """
    Processes images for multi-modal RAG.
    Handles image loading, resizing, and metadata extraction.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    
    # Maximum dimensions for processing
    MAX_DIMENSION = 1024
    THUMBNAIL_SIZE = (256, 256)
    
    def __init__(self, enable_ocr: bool = False):
        """
        Initialize image processor.
        
        Args:
            enable_ocr: Whether to enable OCR for text extraction
        """
        if not PILLOW_AVAILABLE:
            raise ImportError("Pillow is required for ImageProcessor. Install with: pip install Pillow")
        
        self.enable_ocr = enable_ocr
        self._ocr_engine = None
        
        logger.info(f"ImageProcessor initialized (OCR: {enable_ocr})")
    
    def process_image(
        self, 
        image_path: str,
        extract_text: bool = True
    ) -> ProcessedImage:
        """
        Process an image file.
        
        Args:
            image_path: Path to image file
            extract_text: Whether to attempt text extraction
            
        Returns:
            ProcessedImage with metadata and optional extracted content
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        # Load image
        with Image.open(path) as img:
            # Generate image ID
            image_id = self._generate_image_id(path)
            
            # Get basic metadata
            width, height = img.size
            format_str = img.format or path.suffix[1:].upper()
            file_size = path.stat().st_size
            
            # Create thumbnail
            thumbnail_b64 = self._create_thumbnail(img)
            
            # Extract text if enabled
            extracted_text = None
            if extract_text and self.enable_ocr:
                extracted_text = self._extract_text(img)
            
            # Detect objects (basic detection based on image characteristics)
            detected_objects = self._detect_objects(img, path.stem)
            
            # Additional metadata
            metadata = self._extract_metadata(img, path)
            
            return ProcessedImage(
                image_id=image_id,
                width=width,
                height=height,
                format=format_str,
                file_size=file_size,
                path=str(path.absolute()),
                base64_thumbnail=thumbnail_b64,
                extracted_text=extracted_text,
                detected_objects=detected_objects,
                metadata=metadata
            )
    
    def process_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "unknown"
    ) -> ProcessedImage:
        """
        Process image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            filename: Original filename
            
        Returns:
            ProcessedImage
        """
        # Generate image ID from content
        image_id = hashlib.sha256(image_bytes).hexdigest()[:16]
        
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            format_str = img.format or "UNKNOWN"
            
            thumbnail_b64 = self._create_thumbnail(img)
            detected_objects = self._detect_objects(img, filename)
            
            metadata = {
                "source": "bytes",
                "original_filename": filename
            }
            
            return ProcessedImage(
                image_id=image_id,
                width=width,
                height=height,
                format=format_str,
                file_size=len(image_bytes),
                path=None,
                base64_thumbnail=thumbnail_b64,
                extracted_text=None,
                detected_objects=detected_objects,
                metadata=metadata
            )
    
    def resize_for_embedding(
        self, 
        image_path: str,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Image.Image:
        """
        Resize image for embedding model.
        
        Args:
            image_path: Path to image
            target_size: Target size (width, height)
            
        Returns:
            Resized PIL Image
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize with aspect ratio preservation
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create canvas and center image
            canvas = Image.new("RGB", target_size, (255, 255, 255))
            offset = (
                (target_size[0] - img.size[0]) // 2,
                (target_size[1] - img.size[1]) // 2
            )
            canvas.paste(img, offset)
            
            return canvas
    
    def batch_process(
        self, 
        image_paths: List[str]
    ) -> List[ProcessedImage]:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of ProcessedImage objects
        """
        results = []
        for path in image_paths:
            try:
                processed = self.process_image(path)
                results.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
        
        return results
    
    def _generate_image_id(self, path: Path) -> str:
        """Generate unique image ID from path and content."""
        # Use path + file size + modification time for ID
        stat = path.stat()
        content = f"{path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _create_thumbnail(self, img: Image.Image) -> str:
        """Create base64 thumbnail."""
        # Create copy for thumbnail
        thumb = img.copy()
        thumb.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")
        
        # Save to bytes
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _extract_text(self, img: Image.Image) -> Optional[str]:
        """Extract text from image using OCR."""
        # Placeholder - would integrate with pytesseract or similar
        logger.debug("OCR text extraction not implemented")
        return None
    
    def _detect_objects(self, img: Image.Image, filename: str) -> List[str]:
        """
        Basic object detection based on image characteristics and filename.
        For production, integrate with vision model.
        """
        detected = []
        
        # Detect from filename
        filename_lower = filename.lower()
        
        # JD Jones specific detection
        if any(x in filename_lower for x in ["gasket", "seal"]):
            detected.append("gasket")
        if any(x in filename_lower for x in ["packing", "ring"]):
            detected.append("packing")
        if any(x in filename_lower for x in ["valve", "flange"]):
            detected.append("valve_component")
        if any(x in filename_lower for x in ["diagram", "schematic", "drawing"]):
            detected.append("technical_diagram")
        if any(x in filename_lower for x in ["table", "spec", "chart"]):
            detected.append("specification_table")
        if any(x in filename_lower for x in ["photo", "image", "product"]):
            detected.append("product_photo")
        
        # Detect from image characteristics
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            detected.append("banner_image")
        
        # Check if likely a diagram (high contrast, few colors)
        if img.mode == "RGB":
            colors = img.getcolors(256)
            if colors and len(colors) < 20:
                detected.append("diagram")
        
        return detected
    
    def _extract_metadata(self, img: Image.Image, path: Path) -> Dict[str, Any]:
        """Extract image metadata."""
        metadata = {
            "mode": img.mode,
            "filename": path.name,
            "file_extension": path.suffix.lower(),
        }
        
        # EXIF data if available
        try:
            exif = img._getexif()
            if exif:
                metadata["has_exif"] = True
        except Exception:
            metadata["has_exif"] = False
        
        # Image info
        if hasattr(img, "info"):
            for key in ["dpi", "resolution", "description"]:
                if key in img.info:
                    metadata[key] = img.info[key]
        
        return metadata
    
    def image_to_description(self, processed: ProcessedImage) -> str:
        """
        Generate text description of image for indexing.
        
        Args:
            processed: ProcessedImage object
            
        Returns:
            Text description
        """
        parts = []
        
        # Basic info
        parts.append(f"Image: {processed.metadata.get('filename', 'unknown')}")
        parts.append(f"Size: {processed.width}x{processed.height}")
        parts.append(f"Format: {processed.format}")
        
        # Detected objects
        if processed.detected_objects:
            parts.append(f"Contains: {', '.join(processed.detected_objects)}")
        
        # Extracted text
        if processed.extracted_text:
            parts.append(f"Text: {processed.extracted_text[:200]}")
        
        return " | ".join(parts)
