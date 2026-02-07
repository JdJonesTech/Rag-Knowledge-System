"""
Quotation PDF Generator - Generate professional quotation PDFs.
Based on the standard JD Jones quotation format (66361.docx).
"""

import io
from datetime import datetime
from typing import Optional
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from src.quotation.models import QuotationRequest, QuotationLineItem

logger = logging.getLogger(__name__)


class QuotationPDFGenerator:
    """
    Generates professional quotation PDFs in JD Jones corporate format.
    
    IMPORTANT: This is for INTERNAL USE ONLY. Never expose generated PDFs
    with pricing to external portals.
    """
    
    # Company information
    COMPANY_NAME = "J. D. JONES (PACMAAN) PRIVATE LIMITED"
    COMPANY_ADDRESS = """Village Nayla, Nr. Radhaswami Satsang, Ambala Road,
Chandigarh - 160 101 (INDIA)
Phone: +91-172-2652421, 2652422
Email: export@jdjones.com
Website: www.jdjones.com"""
    
    # GST Rate
    GST_RATE = 0.18  # 18% GST
    
    # Placeholder price for products without pricing data
    PLACEHOLDER_PRICE = 500.0
    
    def __init__(self):
        """Initialize PDF generator."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CompanyName',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#8B0000'),
            alignment=TA_CENTER,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='CompanyAddress',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.gray,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='QuotationTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=10,
            textColor=colors.HexColor('#8B0000'),
            spaceBefore=12,
            spaceAfter=6
        ))
    
    def generate_quotation_pdf(
        self,
        request: QuotationRequest,
        include_pricing: bool = True
    ) -> bytes:
        """
        Generate a quotation PDF.
        
        Args:
            request: QuotationRequest with all details
            include_pricing: If True, include pricing (internal only).
                           If False, only include product details (for preview).
        
        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=15*mm,
            bottomMargin=20*mm
        )
        
        story = []
        
        # Header
        story.extend(self._build_header(request))
        
        # Customer details
        story.extend(self._build_customer_section(request))
        
        # Reference section
        story.extend(self._build_reference_section(request))
        
        # Product description
        story.extend(self._build_product_description(request))
        
        # Products table
        story.extend(self._build_products_table(request, include_pricing))
        
        # Totals (if pricing included)
        if include_pricing:
            story.extend(self._build_totals_section(request))
        
        # Terms and conditions
        story.extend(self._build_terms_section())
        
        # Footer
        story.extend(self._build_footer())
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _build_header(self, request: QuotationRequest) -> list:
        """Build the header section."""
        elements = []
        
        # Company name
        elements.append(Paragraph(self.COMPANY_NAME, self.styles['CompanyName']))
        
        # Company address
        elements.append(Paragraph(
            self.COMPANY_ADDRESS.replace('\n', '<br/>'),
            self.styles['CompanyAddress']
        ))
        
        # Quotation header table
        quotation_date = request.quoted_at or datetime.now()
        header_data = [
            [f'Quotation No. {request.quotation_number or "DRAFT"}',
             f'Date: {quotation_date.strftime("%d %B, %Y")}']
        ]
        
        header_table = Table(header_data, colWidths=[90*mm, 80*mm])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#8B0000')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(header_table)
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_customer_section(self, request: QuotationRequest) -> list:
        """Build customer details section."""
        elements = []
        
        if request.customer:
            customer_data = [
                [request.customer.name],
                [request.customer.designation or ''],
                [request.customer.company],
                [request.customer.address or ''],
                [f'Email: {request.customer.email}'],
            ]
            
            # Filter out empty rows
            customer_data = [[row[0]] for row in customer_data if row[0]]
            
            customer_table = Table(customer_data, colWidths=[170*mm])
            customer_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            
            elements.append(customer_table)
            elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_reference_section(self, request: QuotationRequest) -> list:
        """Build reference section."""
        elements = []
        
        elements.append(Paragraph("Dear Sir,", self.styles['Normal']))
        elements.append(Spacer(1, 6))
        
        if request.reference_rfq:
            rfq_date = request.rfq_date.strftime("%d.%m.%Y") if request.rfq_date else "N/A"
            due_date = request.due_date.strftime("%d.%m.%Y") if request.due_date else "N/A"
            ref_text = f"Ref: Your RFQ No. {request.reference_rfq} dated {rfq_date}."
            if request.due_date:
                ref_text += f" Due date is on {due_date}."
            elements.append(Paragraph(ref_text, self.styles['Normal']))
        
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "We acknowledge receipt of your above enquiry and are pleased to submit our offer as under.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_product_description(self, request: QuotationRequest) -> list:
        """Build product description section."""
        elements = []
        
        elements.append(Paragraph("JDJ Product Description:", self.styles['SectionHeader']))
        
        # Group items by product code and describe
        product_descriptions = {}
        for item in request.line_items:
            if item.product_code not in product_descriptions:
                product_descriptions[item.product_code] = item.product_name
        
        for code, name in product_descriptions.items():
            elements.append(Paragraph(
                f'# "{code}" - {name}',
                self.styles['Normal']
            ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_products_table(self, request: QuotationRequest, include_pricing: bool) -> list:
        """Build products table."""
        elements = []
        
        # Determine dimension unit from line items (must be before header construction)
        dim_units_used = set()
        for item in request.line_items:
            dim_units_used.add(getattr(item, 'dimension_unit', 'mm') or 'mm')
        if len(dim_units_used) == 1:
            dim_unit_label = next(iter(dim_units_used))
        elif len(dim_units_used) > 1:
            dim_unit_label = 'mm/inch'
        else:
            dim_unit_label = 'mm'
        mixed_dim_units = len(dim_units_used) > 1
        
        # Table headers
        if include_pricing:
            headers = ['Sl.', 'JDJ Style', 'Material Code', f'Size ({dim_unit_label})\nOD x ID x TH',
                      'Rings/Set', 'Qty', 'Unit', 'Price/UOM\n(Rs.)', 'Amount\nExcl. GST\n(Rs.)']
            col_widths = [10*mm, 20*mm, 35*mm, 30*mm, 18*mm, 15*mm, 15*mm, 22*mm, 25*mm]
        else:
            headers = ['Sl.', 'JDJ Style', 'Material Code', f'Size ({dim_unit_label})\nOD x ID x TH',
                      'Rings/Set', 'Qty', 'Unit', 'Notes']
            col_widths = [12*mm, 25*mm, 40*mm, 35*mm, 20*mm, 18*mm, 20*mm, 35*mm]
        
        data = [headers]
        
        
        for idx, item in enumerate(request.line_items, 1):
            item_unit = getattr(item, 'dimension_unit', 'mm') or 'mm'
            if mixed_dim_units:
                od_str = f"{item.size_od} {item_unit}" if item.size_od else '-'
                id_str = f"{item.size_id} {item_unit}" if item.size_id else '-'
                th_str = f"{item.size_th} {item_unit}" if item.size_th else '-'
                size_str = f"{od_str} x {id_str} x {th_str}"
            else:
                size_str = f"{item.size_od or '-'} x {item.size_id or '-'} x {item.size_th or '-'}"
            
            if include_pricing:
                unit_price = item.unit_price or self.PLACEHOLDER_PRICE
                amount = unit_price * item.quantity
                row = [
                    str(idx).zfill(2),
                    item.product_code,
                    item.material_code or '-',
                    size_str,
                    str(item.rings_per_set) if item.rings_per_set else '-',
                    str(item.quantity),
                    item.unit,
                    f'{unit_price:,.0f}',
                    f'{amount:,.0f}'
                ]
            else:
                row = [
                    str(idx).zfill(2),
                    item.product_code,
                    item.material_code or '-',
                    size_str,
                    str(item.rings_per_set) if item.rings_per_set else '-',
                    str(item.quantity),
                    item.unit,
                    item.notes or '-'
                ]
            
            data.append(row)
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            # Header style
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8B0000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            
            # Data style
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Sl. No
            ('ALIGN', (4, 1), (-1, -1), 'CENTER'),  # Numeric columns
            ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#8B0000')),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_totals_section(self, request: QuotationRequest) -> list:
        """Build totals section (internal only)."""
        elements = []
        
        # Calculate totals
        subtotal = 0.0
        for item in request.line_items:
            unit_price = item.unit_price or self.PLACEHOLDER_PRICE
            subtotal += unit_price * item.quantity
        
        gst = subtotal * self.GST_RATE
        total = subtotal + gst
        
        totals_data = [
            ['', '', '', '', '', '', '', 'Sub Total:', f'{subtotal:,.0f}'],
            ['', '', '', '', '', '', '', f'GST @ {int(self.GST_RATE*100)}%:', f'{gst:,.0f}'],
            ['', '', '', '', '', '', '', 'Grand Total:', f'{total:,.0f}'],
        ]
        
        totals_table = Table(totals_data, colWidths=[10*mm, 20*mm, 35*mm, 30*mm, 18*mm, 15*mm, 15*mm, 22*mm, 25*mm])
        totals_table.setStyle(TableStyle([
            ('FONTNAME', (7, 0), (7, -1), 'Helvetica-Bold'),
            ('FONTNAME', (8, 0), (8, -1), 'Helvetica-Bold'),
            ('ALIGN', (7, 0), (-1, -1), 'RIGHT'),
            ('LINEABOVE', (7, 0), (-1, 0), 1, colors.black),
            ('LINEBELOW', (7, -1), (-1, -1), 2, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(totals_table)
        elements.append(Spacer(1, 12))
        
        # Note about placeholder pricing
        elements.append(Paragraph(
            "<i>Note: Prices shown are indicative. Final pricing will be confirmed by sales team.</i>",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_terms_section(self) -> list:
        """Build terms and conditions section."""
        elements = []
        
        elements.append(Paragraph("Terms & Conditions:", self.styles['SectionHeader']))
        
        terms = [
            "1. Delivery: Ex-Works, Chandigarh",
            "2. Payment: 100% advance against Proforma Invoice",
            "3. Validity: This quotation is valid for 30 days from date of issue",
            "4. Taxes: GST @ 18% extra as applicable",
            "5. Packing: Standard export packing included",
        ]
        
        for term in terms:
            elements.append(Paragraph(term, self.styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_footer(self) -> list:
        """Build footer section."""
        elements = []
        
        elements.append(Paragraph(
            "We trust the above is in line with your requirement and look forward to receiving your valued order.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        elements.append(Paragraph("Thanking you,", self.styles['Normal']))
        elements.append(Paragraph("Yours faithfully,", self.styles['Normal']))
        elements.append(Paragraph("<b>For J. D. JONES (PACMAAN) PVT. LTD.</b>", self.styles['Normal']))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Authorized Signatory", self.styles['Normal']))
        
        return elements


# Singleton instance
_pdf_generator: Optional[QuotationPDFGenerator] = None


def get_pdf_generator() -> QuotationPDFGenerator:
    """Get singleton PDF generator instance."""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = QuotationPDFGenerator()
    return _pdf_generator
