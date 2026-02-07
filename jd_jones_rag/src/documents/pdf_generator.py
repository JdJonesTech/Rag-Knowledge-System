"""
PDF Document Generator
Generates professional PDF documents for quotations, datasheets, and proposals.
Uses HTML templates with Jinja2 and weasyprint for PDF rendering.
Falls back to simple PDF generation if weasyprint is unavailable.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents that can be generated."""
    QUOTATION = "quotation"
    DATASHEET = "datasheet"
    PROPOSAL = "proposal"
    SPEC_SHEET = "spec_sheet"
    CERTIFICATE = "certificate"


@dataclass
class GeneratedDocument:
    """Represents a generated document."""
    doc_id: str
    doc_type: DocumentType
    title: str
    filename: str
    file_path: str
    format: str  # pdf, html, markdown
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "title": self.title,
            "filename": self.filename,
            "file_path": self.file_path,
            "format": self.format,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


class PDFGenerator:
    """
    Generates PDF documents from product and enquiry data.
    """
    
    def __init__(self, output_dir: str = "data/output/documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir = Path("src/templates/documents")
        
        # Check if weasyprint is available
        try:
            from weasyprint import HTML, CSS
            self.weasyprint_available = True
            logger.info("WeasyPrint available for PDF generation")
        except (ImportError, OSError, Exception):
            self.weasyprint_available = False
            logger.warning("WeasyPrint not available, falling back to reportlab")
        
        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            self.reportlab_available = True
            logger.info("ReportLab available for PDF generation")
        except ImportError:
            self.reportlab_available = False
            logger.warning("ReportLab not available")
    
    def generate_quotation(
        self,
        customer_name: str,
        customer_email: str,
        products: List[Dict[str, Any]],
        notes: str = "",
        validity_days: int = 30,
        terms: str = "Standard terms and conditions apply",
        **kwargs
    ) -> GeneratedDocument:
        """Generate a quotation PDF."""
        
        doc_id = f"QUO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Calculate totals
        subtotal = sum(float(p.get('unit_price', 0)) * int(p.get('quantity', 1)) for p in products)
        tax = subtotal * 0.18  # 18% GST
        total = subtotal + tax
        
        content = {
            "doc_id": doc_id,
            "date": datetime.now().strftime("%d %B %Y"),
            "valid_until": (datetime.now() + timedelta(days=validity_days)).strftime("%d %B %Y"),
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_company": kwargs.get("customer_company", ""),
            "customer_designation": kwargs.get("customer_designation", ""),
            "customer_address": kwargs.get("customer_address", ""),
            "rfq_number": kwargs.get("rfq_number", ""),
            "rfq_date": kwargs.get("rfq_date", ""),
            "due_date": kwargs.get("due_date", ""),
            "products": products,
            "subtotal": f"Rs. {subtotal:,.0f}",
            "tax": f"Rs. {tax:,.0f}",
            "total": f"Rs. {total:,.0f}",
            "notes": notes,
            "terms": terms
        }
        
        filename = f"{doc_id}.pdf"
        file_path = self.output_dir / filename
        
        if self.reportlab_available:
            self._generate_quotation_reportlab(content, file_path)
            format_type = "pdf"
        else:
            # Fallback to HTML
            html_path = file_path.with_suffix('.html')
            self._generate_quotation_html(content, html_path)
            file_path = html_path
            filename = html_path.name
            format_type = "html"
        
        return GeneratedDocument(
            doc_id=doc_id,
            doc_type=DocumentType.QUOTATION,
            title=f"Quotation for {customer_name}",
            filename=filename,
            file_path=str(file_path),
            format=format_type,
            created_at=datetime.now(),
            metadata=content
        )
    
    def _generate_quotation_reportlab(self, content: Dict[str, Any], output_path: Path):
        """Generate quotation PDF using ReportLab - matching 66361.docx format exactly."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.units import mm, cm
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        
        PAGE_WIDTH, PAGE_HEIGHT = A4
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=14*mm,
            leftMargin=25*mm,
            topMargin=15*mm,
            bottomMargin=20*mm
        )
        styles = getSampleStyleSheet()
        story = []
        
        # --- Custom styles ---
        company_style = ParagraphStyle(
            'CompanyName', parent=styles['Heading1'],
            fontSize=14, textColor=colors.HexColor('#000000'),
            alignment=TA_CENTER, spaceAfter=2
        )
        address_style = ParagraphStyle(
            'CompanyAddr', parent=styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=colors.HexColor('#444444'), spaceAfter=8
        )
        section_bold = ParagraphStyle(
            'SectionBold', parent=styles['Normal'],
            fontSize=10, fontName='Helvetica-Bold', spaceAfter=4
        )
        normal_style = ParagraphStyle(
            'NormalQ', parent=styles['Normal'],
            fontSize=10, spaceAfter=4
        )
        small_style = ParagraphStyle(
            'SmallQ', parent=styles['Normal'],
            fontSize=9, spaceAfter=2
        )
        center_style = ParagraphStyle(
            'CenterQ', parent=styles['Normal'],
            fontSize=9, alignment=TA_CENTER, spaceAfter=2
        )
        
        # ==========================================
        # 1. HEADER - Company Logo
        # ==========================================
        logo_path = Path(__file__).parent.parent / "quotation" / "jdj_logo.png"
        if logo_path.exists():
            # Scale logo to fit page width (approx 170mm usable)
            logo_img = RLImage(str(logo_path), width=165*mm, height=37*mm)
            story.append(logo_img)
            story.append(Spacer(1, 4*mm))
        else:
            story.append(Paragraph("J D Jones & Co (P) Ltd", company_style))
            story.append(Paragraph(
                "Village Nayla, Nr. Radhaswami Satsang, Ambala Road, "
                "Chandigarh - 160 101 (INDIA)<br/>"
                "Phone: +91-172-2652421, 2652422 | "
                "Email: export@jdjones.com | Website: www.jdjones.com",
                address_style
            ))
        
        # ==========================================
        # 2. QUOTATION NUMBER + DATE (Table 0 in docx)
        # ==========================================
        quo_num = content.get('doc_id', 'DRAFT')
        quo_date = content.get('date', datetime.now().strftime("%d %B, %Y"))
        
        header_data = [[
            Paragraph(f'<b>Quotation No. {quo_num}</b>', small_style),
            Paragraph(f'<b>Date: {quo_date}</b>', ParagraphStyle(
                'DateR', parent=small_style, alignment=TA_RIGHT
            ))
        ]]
        header_tbl = Table(header_data, colWidths=[90*mm, 75*mm])
        header_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F0F0')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 4*mm))
        
        # ==========================================
        # 3. CUSTOMER DETAILS (Table 1 in docx)
        # ==========================================
        cust_name = content.get('customer_name', 'Customer')
        cust_email = content.get('customer_email', '')
        cust_company = content.get('customer_company', '')
        cust_designation = content.get('customer_designation', '')
        cust_address = content.get('customer_address', '')
        
        cust_rows = []
        cust_rows.append([Paragraph(f'<b>{cust_name}</b>', small_style)])
        if cust_designation:
            cust_rows.append([Paragraph(cust_designation, small_style)])
        if cust_company:
            cust_rows.append([Paragraph(cust_company, small_style)])
        if cust_address:
            cust_rows.append([Paragraph(cust_address, small_style)])
        if cust_email:
            cust_rows.append([Paragraph(f'Email: {cust_email}', small_style)])
        
        if cust_rows:
            cust_tbl = Table(cust_rows, colWidths=[165*mm])
            cust_tbl.setStyle(TableStyle([
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(cust_tbl)
            story.append(Spacer(1, 4*mm))
        
        # ==========================================
        # 4. REFERENCE SECTION
        # ==========================================
        story.append(Paragraph("Dear Sir,", normal_style))
        story.append(Spacer(1, 2*mm))
        
        rfq = content.get('rfq_number', '')
        rfq_date = content.get('rfq_date', '')
        due_date = content.get('due_date', '')
        if rfq:
            ref_text = f"<b>Ref: Your RFQ No. {rfq}"
            if rfq_date:
                ref_text += f" dated {rfq_date}"
            ref_text += "."
            if due_date:
                ref_text += f" Due date is on {due_date}."
            ref_text += "</b>"
            story.append(Paragraph(ref_text, normal_style))
            story.append(Spacer(1, 2*mm))
        
        story.append(Paragraph(
            "<b>We acknowledge receipt of your above enquiry and are pleased to submit our offer as under.</b>",
            normal_style
        ))
        story.append(Spacer(1, 3*mm))
        
        # ==========================================
        # 5. PRODUCT DESCRIPTIONS
        # ==========================================
        story.append(Paragraph("<b>JDJ Product Description:</b>", normal_style))
        story.append(Spacer(1, 2*mm))
        
        # Group products by code and show descriptions
        seen_codes = set()
        for p in content.get('products', []):
            code = p.get('code', '')
            if code and code not in seen_codes:
                seen_codes.add(code)
                name = p.get('name', code)
                grade = p.get('material_grade', '')
                desc = f'<b># "Pacmaan" Style {code} {name}</b>'
                if grade and grade != '-':
                    desc = f'<b># "Pacmaan" Style {code} {name} - {grade}</b>'
                story.append(Paragraph(desc, small_style))
        story.append(Spacer(1, 4*mm))
        
        # ==========================================
        # 6. PRODUCTS TABLE (matching 66361.docx Table 2)
        # ==========================================
        # Column widths (total ~165mm usable)
        col_widths = [
            10*mm,   # Sl. No.
            20*mm,   # JDJ Style
            32*mm,   # Material Code
            14*mm,   # OD
            14*mm,   # ID
            12*mm,   # TH
            14*mm,   # Rings/Set
            12*mm,   # Qty
            14*mm,   # Unit
            18*mm,   # Price/UOM
            18*mm,   # Amount
        ]
        
        # Header row 1 (with merged "Size (in mm)" spanning OD, ID, TH)
        hdr_cell_style = ParagraphStyle(
            'HdrCell', parent=styles['Normal'],
            fontSize=7, fontName='Helvetica-Bold',
            alignment=TA_CENTER, textColor=colors.white
        )
        
        # Determine dimension unit label from products
        dim_units_used = set()
        for p in content.get('products', []):
            dim_units_used.add(p.get('dimension_unit', 'mm'))
        if len(dim_units_used) == 1:
            dim_unit_label = f"Size (in {dim_units_used.pop()})"
            mixed_units = False
        elif len(dim_units_used) > 1:
            dim_unit_label = "Size (mm/inch)"
            mixed_units = True
        else:
            dim_unit_label = "Size (in mm)"
            mixed_units = False
        
        header_row1 = [
            Paragraph('Sl.<br/>No.', hdr_cell_style),
            Paragraph('JDJ<br/>Style', hdr_cell_style),
            Paragraph('Material<br/>Code', hdr_cell_style),
            Paragraph(dim_unit_label, hdr_cell_style),  # spans 3 cols
            '', '',  # merged cells
            Paragraph('Rings/<br/>Set', hdr_cell_style),
            Paragraph('Qty', hdr_cell_style),
            Paragraph('Unit', hdr_cell_style),
            Paragraph('Price/UOM<br/>(Rs.)', hdr_cell_style),
            Paragraph('Amount<br/>Excl. GST<br/>(Rs.)', hdr_cell_style),
        ]
        
        header_row2 = [
            '', '', '',  # merged from row 1
            Paragraph('OD', hdr_cell_style),
            Paragraph('ID', hdr_cell_style),
            Paragraph('TH', hdr_cell_style),
            '', '', '', '', '',  # merged from row 1
        ]
        
        # Data rows
        data_cell_style = ParagraphStyle(
            'DataCell', parent=styles['Normal'],
            fontSize=8, alignment=TA_CENTER
        )
        data_cell_left = ParagraphStyle(
            'DataCellL', parent=styles['Normal'],
            fontSize=8, alignment=TA_LEFT
        )
        
        data_rows = []
        subtotal = 0.0
        
        for idx, p in enumerate(content.get('products', []), 1):
            qty = int(p.get('quantity', 1))
            price = float(p.get('unit_price', 0))
            amount = qty * price
            subtotal += amount
            
            # Parse size components
            size = p.get('size', '-')
            od, id_val, th = '-', '-', '-'
            if isinstance(size, str) and 'x' in size:
                parts = [s.strip() for s in size.split('x')]
                if len(parts) >= 1: od = parts[0]
                if len(parts) >= 2: id_val = parts[1]
                if len(parts) >= 3: th = parts[2]
            elif isinstance(size, dict):
                od = str(size.get('od', '-'))
                id_val = str(size.get('id', '-'))
                th = str(size.get('th', '-'))
            # Build material description - show code and grade
            mat_code = p.get('material_code', '')
            mat_grade = p.get('material_grade', '')
            mat_display = p.get('material', '-')
            if mat_code and mat_grade and mat_code != mat_grade and mat_grade != '-':
                mat_display = f'{mat_code}<br/><font size="6">({mat_grade})</font>'
            elif mat_code:
                mat_display = mat_code
            elif mat_grade and mat_grade != '-':
                mat_display = mat_grade
            
            # Get per-item dimension unit for display
            item_dim_unit = p.get('dimension_unit', 'mm')
            
            # If mixed units, append unit suffix to each dimension value
            if mixed_units:
                od_display = f'{od} {item_dim_unit}' if od != '-' else od
                id_display = f'{id_val} {item_dim_unit}' if id_val != '-' else id_val
                th_display = f'{th} {item_dim_unit}' if th != '-' else th
            else:
                od_display = str(od)
                id_display = str(id_val)
                th_display = str(th)
            
            row = [
                Paragraph(str(idx).zfill(2), data_cell_style),
                Paragraph(p.get('code', '-'), data_cell_style),
                Paragraph(mat_display, data_cell_left),
                Paragraph(od_display, data_cell_style),
                Paragraph(id_display, data_cell_style),
                Paragraph(th_display, data_cell_style),
                Paragraph(str(p.get('rings_per_set', '-')), data_cell_style),
                Paragraph(str(qty), data_cell_style),
                Paragraph(p.get('unit', 'Nos.'), data_cell_style),
                Paragraph(f'{price:,.0f}' if price else '-', data_cell_style),
                Paragraph(f'{amount:,.0f}' if amount else '-', data_cell_style),
            ]
            data_rows.append(row)
        
        # Total row
        total_label_style = ParagraphStyle(
            'TotalLabel', parent=styles['Normal'],
            fontSize=8, fontName='Helvetica-Bold', alignment=TA_RIGHT
        )
        total_val_style = ParagraphStyle(
            'TotalVal', parent=styles['Normal'],
            fontSize=8, fontName='Helvetica-Bold', alignment=TA_CENTER
        )
        
        total_row = [
            '', '',
            Paragraph('<b>Total Amount Excl GST (In Rs.)</b>', total_label_style),
            '', '', '', '', '', '', '',
            Paragraph(f'<b>{subtotal:,.0f}</b>', total_val_style),
        ]
        
        # Compose full table data
        table_data = [header_row1, header_row2] + data_rows + [total_row]
        
        num_data_rows = len(data_rows)
        total_row_idx = 2 + num_data_rows  # 0-indexed
        
        products_table = Table(table_data, colWidths=col_widths, repeatRows=2)
        
        table_style_cmds = [
            # Header background
            ('BACKGROUND', (0, 0), (-1, 1), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 1), colors.white),
            
            # Header row 1 merges: Size (in mm) spans cols 3-5
            ('SPAN', (3, 0), (5, 0)),
            # Vertical merges for row 0-1
            ('SPAN', (0, 0), (0, 1)),    # Sl. No.
            ('SPAN', (1, 0), (1, 1)),    # JDJ Style
            ('SPAN', (2, 0), (2, 1)),    # Material Code
            ('SPAN', (6, 0), (6, 1)),    # Rings/Set
            ('SPAN', (7, 0), (7, 1)),    # Qty
            ('SPAN', (8, 0), (8, 1)),    # Unit
            ('SPAN', (9, 0), (9, 1)),    # Price/UOM
            ('SPAN', (10, 0), (10, 1)),  # Amount
            
            # Total row merge: cols 2-9 for label
            ('SPAN', (2, total_row_idx), (9, total_row_idx)),
            
            # Total row background
            ('BACKGROUND', (0, total_row_idx), (-1, total_row_idx), colors.HexColor('#D9E2F3')),
            ('FONTNAME', (0, total_row_idx), (-1, total_row_idx), 'Helvetica-Bold'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#999999')),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('LINEBELOW', (0, 1), (-1, 1), 1, colors.black),
            
            # Alignment
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 1), 'CENTER'),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            
            # Alternate row shading for data rows
        ]
        
        # Add alternating row colors for readability
        for i in range(2, 2 + num_data_rows):
            if i % 2 == 0:
                table_style_cmds.append(
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#FFFFFF'))
                )
            else:
                table_style_cmds.append(
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F2F6FC'))
                )
        
        products_table.setStyle(TableStyle(table_style_cmds))
        story.append(products_table)
        story.append(Spacer(1, 6*mm))
        
        # ==========================================
        # 7. TERMS & CONDITIONS
        # ==========================================
        terms_items = [
            "F.O.R. Destination by road.",
            "",
            "IGST will be extra as applicable at the time of despatch w.e.f. 1st July, 2017. The GST rate is 18%.",
            "Our GST No. 19AAACJ7932R1ZS.",
            "",
            "Payment: 100% within 30 days.",
            "",
            "Delivery: Materials will be delivered within 6 to 8 weeks from the date of receipt of order.",
            "",
            "Validity: This offer is valid for 90 days from the date of our offer and thereafter subject to our confirmation.",
            "",
            "Note: We are registered by Small Category under Enterprises Development Act, 2006 (copy of NSIC certificate enclosed)",
        ]
        
        for term in terms_items:
            if term:
                story.append(Paragraph(f"<b>{term}</b>", small_style))
            else:
                story.append(Spacer(1, 2*mm))
        
        story.append(Spacer(1, 4*mm))
        
        # ==========================================
        # 8. CLOSING / FOOTER
        # ==========================================
        story.append(Paragraph("Thanking you and looking forward to receiving your valued order.", normal_style))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("Yours faithfully,", normal_style))
        story.append(Paragraph("<b>J D Jones & Co (P) Ltd.</b>", normal_style))
        story.append(Spacer(1, 15*mm))
        story.append(Paragraph(
            "(SIGNATURE NOT REQUIRED SINCE TRANSMITTED ELECTRONICALLY)",
            center_style
        ))
        
        doc.build(story)
        logger.info(f"Generated quotation PDF: {output_path}")
    
    def _generate_quotation_html(self, content: Dict[str, Any], output_path: Path):
        """Generate quotation as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quotation {content['doc_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #952825; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th {{ background-color: #952825; color: white; padding: 10px; }}
        td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .total {{ font-size: 1.2em; color: #952825; font-weight: bold; }}
        .footer {{ margin-top: 40px; font-size: 0.8em; color: #666; }}
    </style>
</head>
<body>
    <h1 class="header">JD JONES</h1>
    <p>Sealing Solutions</p>
    <h2>QUOTATION: {content['doc_id']}</h2>
    <p><strong>To:</strong> {content['customer_name']}</p>
    <p><strong>Email:</strong> {content['customer_email']}</p>
    <p><strong>Date:</strong> {content['date']}</p>
    <p><strong>Valid Until:</strong> {content['valid_until']}</p>
    
    <table>
        <tr>
            <th>Product Code</th>
            <th>Description</th>
            <th>Quantity</th>
            <th>Unit Price</th>
            <th>Total</th>
        </tr>
"""
        for p in content['products']:
            qty = int(p.get('quantity', 1))
            price = float(p.get('unit_price', 0))
            total = qty * price
            html += f"""
        <tr>
            <td>{p.get('code', 'N/A')}</td>
            <td>{p.get('name', 'N/A')}</td>
            <td>{qty}</td>
            <td>₹{price:,.2f}</td>
            <td>₹{total:,.2f}</td>
        </tr>
"""
        html += f"""
    </table>
    
    <p><strong>Subtotal:</strong> {content['subtotal']}</p>
    <p><strong>GST (18%):</strong> {content['tax']}</p>
    <p class="total"><strong>TOTAL:</strong> {content['total']}</p>
    
    <p><strong>Notes:</strong> {content['notes']}</p>
    <p><strong>Terms:</strong> {content['terms']}</p>
    
    <div class="footer">
        <p>JD Jones Sealing Solutions | sales@jdjones.com | +91-xxx-xxx-xxxx</p>
    </div>
</body>
</html>
"""
        output_path.write_text(html, encoding='utf-8')
        logger.info(f"Generated quotation HTML: {output_path}")
    
    def generate_datasheet(
        self,
        product_code: str,
        product_name: str,
        specifications: Dict[str, Any],
        certifications: List[str] = None,
        applications: List[str] = None,
        materials: Dict[str, str] = None
    ) -> GeneratedDocument:
        """Generate a product datasheet PDF."""
        
        doc_id = f"DS-{product_code.replace(' ', '')}-{uuid.uuid4().hex[:4].upper()}"
        
        content = {
            "doc_id": doc_id,
            "product_code": product_code,
            "product_name": product_name,
            "specifications": specifications,
            "certifications": certifications or [],
            "applications": applications or [],
            "materials": materials or {},
            "generated_date": datetime.now().strftime("%d %B %Y")
        }
        
        filename = f"{doc_id}.pdf"
        file_path = self.output_dir / filename
        
        if self.reportlab_available:
            self._generate_datasheet_reportlab(content, file_path)
            format_type = "pdf"
        else:
            html_path = file_path.with_suffix('.html')
            self._generate_datasheet_html(content, html_path)
            file_path = html_path
            filename = html_path.name
            format_type = "html"
        
        return GeneratedDocument(
            doc_id=doc_id,
            doc_type=DocumentType.DATASHEET,
            title=f"Datasheet - {product_code} {product_name}",
            filename=filename,
            file_path=str(file_path),
            format=format_type,
            created_at=datetime.now(),
            metadata=content
        )
    
    def _generate_datasheet_reportlab(self, content: Dict[str, Any], output_path: Path):
        """Generate datasheet PDF using ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#952825'),
            spaceAfter=10
        )
        story.append(Paragraph("JD JONES", header_style))
        story.append(Paragraph("Product Datasheet", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Product Info
        product_style = ParagraphStyle(
            'Product',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2d2d2d')
        )
        story.append(Paragraph(f"{content['product_code']}", product_style))
        story.append(Paragraph(f"{content['product_name']}", styles['Heading3']))
        story.append(Spacer(1, 20))
        
        # Specifications Table
        if content['specifications']:
            story.append(Paragraph("<b>Technical Specifications</b>", styles['Heading3']))
            spec_data = [[k, str(v)] for k, v in content['specifications'].items()]
            spec_table = Table(spec_data, colWidths=[6*cm, 10*cm])
            spec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ]))
            story.append(spec_table)
            story.append(Spacer(1, 15))
        
        # Applications
        if content['applications']:
            story.append(Paragraph("<b>Applications</b>", styles['Heading3']))
            for app in content['applications']:
                story.append(Paragraph(f"• {app}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Certifications
        if content['certifications']:
            story.append(Paragraph("<b>Certifications</b>", styles['Heading3']))
            for cert in content['certifications']:
                story.append(Paragraph(f"[OK] {cert}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Materials
        if content['materials']:
            story.append(Paragraph("<b>Materials</b>", styles['Heading3']))
            for component, material in content['materials'].items():
                story.append(Paragraph(f"• {component}: {material}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 40))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)
        story.append(Paragraph(f"Document ID: {content['doc_id']} | Generated: {content['generated_date']}", footer_style))
        story.append(Paragraph("JD Jones Sealing Solutions | www.jdjones.com", footer_style))
        
        doc.build(story)
        logger.info(f"Generated datasheet PDF: {output_path}")
    
    def _generate_datasheet_html(self, content: Dict[str, Any], output_path: Path):
        """Generate datasheet as HTML."""
        specs_html = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in content['specifications'].items())
        apps_html = "".join(f"<li>{app}</li>" for app in content['applications'])
        certs_html = "".join(f"<li>[OK] {cert}</li>" for cert in content['certifications'])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Datasheet - {content['product_code']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #952825; }}
        .product-code {{ font-size: 2em; color: #2d2d2d; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        td {{ border: 1px solid #ddd; padding: 10px; }}
        td:first-child {{ background-color: #f5f5f5; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
        .footer {{ margin-top: 40px; font-size: 0.8em; color: #666; }}
    </style>
</head>
<body>
    <h1 class="header">JD JONES</h1>
    <h2>Product Datasheet</h2>
    <p class="product-code">{content['product_code']}</p>
    <h3>{content['product_name']}</h3>
    
    <div class="section">
        <h3>Technical Specifications</h3>
        <table>{specs_html}</table>
    </div>
    
    <div class="section">
        <h3>Applications</h3>
        <ul>{apps_html}</ul>
    </div>
    
    <div class="section">
        <h3>Certifications</h3>
        <ul>{certs_html}</ul>
    </div>
    
    <div class="footer">
        <p>Document ID: {content['doc_id']} | Generated: {content['generated_date']}</p>
        <p>JD Jones Sealing Solutions | www.jdjones.com</p>
    </div>
</body>
</html>
"""
        output_path.write_text(html, encoding='utf-8')
        logger.info(f"Generated datasheet HTML: {output_path}")
    
    def list_generated_documents(self) -> List[Dict[str, Any]]:
        """List all generated documents."""
        documents = []
        for file in self.output_dir.glob("*"):
            if file.is_file():
                documents.append({
                    "filename": file.name,
                    "path": str(file),
                    "size_bytes": file.stat().st_size,
                    "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
                })
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Path]:
        """Get path to a generated document by ID."""
        for file in self.output_dir.glob(f"{doc_id}.*"):
            return file
        return None


# Required import for quotation validity calculation
from datetime import timedelta

# Global instance
_pdf_generator: Optional[PDFGenerator] = None


def get_pdf_generator() -> PDFGenerator:
    """Get or create global PDF generator instance."""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGenerator()
    return _pdf_generator
