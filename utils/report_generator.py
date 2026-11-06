"""
Diabetic Retinopathy Detection - PDF Medical Report Generator
Generates clinical diagnostic reports using ReportLab.
Includes:
- Clinic branding & report ID
- Patient / Scan metadata
- Original & enhanced retinal fundus images
- Model prediction summary & confidence metrics
- Class probability breakdown
- Clinical findings & recommended next steps
- Required medical disclaimer & digital sign-off block
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)


def generate_pdf_report(
    output_pdf_path,
    scan_id,
    user_name,
    original_img_path,
    enhanced_img_path,
    prediction_data,
    scan_date=None
):
    """
    Generates a clean PDF diagnostic report.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    if not scan_date:
        scan_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b')
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor('#b91c1c'),
        alignment=1
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>RETINANOVA AI</b><br/><font size=8 color='#2563eb'>Hybrid Deep Learning Retinal Diagnostic System</font>", title_style),
            Paragraph(f"<b>REPORT ID:</b> #{scan_id}<br/><b>DATE:</b> {scan_date}<br/><b>STATUS:</b> Completed", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 190])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=10))

    # 2. Patient / User & Scan Metadata Table
    meta_data = [
        [
            Paragraph("<b>Patient / Operator:</b>", bold_body),
            Paragraph(str(user_name), body_style),
            Paragraph("<b>Model Architecture:</b>", bold_body),
            Paragraph("Hybrid CNN + EfficientNet", body_style)
        ],
        [
            Paragraph("<b>Analysis Modality:</b>", bold_body),
            Paragraph("Color Digital Retinal Fundus", body_style),
            Paragraph("<b>Enhancement:</b>", bold_body),
            Paragraph("Ben Graham Local Contrast Filter", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[120, 150, 120, 140])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Retinal Images (Original and Ben Graham Contrast Enhanced)
    img_cells = []
    has_orig = original_img_path and os.path.exists(original_img_path)
    has_enh = enhanced_img_path and os.path.exists(enhanced_img_path)

    if has_orig and has_enh:
        img_cells = [
            [
                RLImage(original_img_path, width=170, height=170),
                RLImage(enhanced_img_path, width=170, height=170)
            ],
            [
                Paragraph("<b>Figure A:</b> Original Retinal Fundus", subtitle_style),
                Paragraph("<b>Figure B:</b> Enhanced Vascular / Lesion Filter", subtitle_style)
            ]
        ]
        img_table = Table(img_cells, colWidths=[265, 265])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 10))

    # 4. Diagnostic Result Box
    stage_name = prediction_data.get('stage_name', 'Unknown')
    pred_stage = prediction_data.get('predicted_stage', '')
    risk_level = prediction_data.get('risk_level', 'Pending')
    confidence = prediction_data.get('confidence', 0.0)

    color_hex = prediction_data.get('color', '#2563eb')
    badge_bg = colors.HexColor(color_hex)

    result_box_data = [
        [
            Paragraph(f"<font color='white'><b>PRIMARY DIAGNOSIS: {pred_stage} - {stage_name.upper()}</b></font>", bold_body),
            Paragraph(f"<font color='white'><b>RISK LEVEL: {risk_level.upper()}</b></font>", bold_body)
        ],
        [
            Paragraph(f"<b>Overall Model Confidence:</b> {confidence}%", bold_body),
            Paragraph(f"<b>Urgency:</b> {prediction_data.get('urgency', 'Routine')}", bold_body)
        ]
    ]
    result_table = Table(result_box_data, colWidths=[320, 210])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), badge_bg),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, badge_bg),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 10))

    # 5. Class Probabilities Table
    story.append(Paragraph("<b>Class-wise Softmax Prediction Probabilities</b>", h2_style))
    probs = prediction_data.get('probabilities', [0, 0, 0, 0, 0])
    prob_headers = ["Class 0: No DR", "Class 1: Mild", "Class 2: Moderate", "Class 3: Severe", "Class 4: Proliferative"]
    prob_row = [f"{p:.2f}%" for p in probs]
    
    prob_table_data = [
        [Paragraph(f"<b>{h}</b>", subtitle_style) for h in prob_headers],
        [Paragraph(f"<b>{v}</b>", body_style) for v in prob_row]
    ]
    prob_table = Table(prob_table_data, colWidths=[106, 106, 106, 106, 106])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # 6. Clinical Description & Findings
    story.append(Paragraph("<b>Clinical Evaluation & Pathological Indicators</b>", h2_style))
    story.append(Paragraph(prediction_data.get('description', ''), body_style))
    story.append(Spacer(1, 4))

    findings = prediction_data.get('clinical_findings', [])
    for f in findings:
        story.append(Paragraph(f"&bull; {f}", body_style))
    story.append(Spacer(1, 8))

    # 7. Recommended Next Steps
    story.append(Paragraph("<b>Suggested Clinical Action & Management Plan</b>", h2_style))
    recs = prediction_data.get('recommendations', [])
    for idx, r in enumerate(recs, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {r}", body_style))
    story.append(Spacer(1, 12))

    # 8. Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#ef4444'), spaceAfter=6))
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> This application is an educational and research prototype. "
        "It is not a certified medical diagnostic tool. Predictions must not replace professional examination "
        "by a qualified ophthalmologist. All clinical decisions must be confirmed by board-certified eye specialists."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 8))

    # 9. Sign-off Footer
    sign_data = [
        [
            Paragraph("<b>Automated Screening Signature:</b> RetinaNova AI Engine v2.4", subtitle_style),
            Paragraph("<b>Ophthalmologist Review Sign-off:</b> ____________________", subtitle_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[280, 250])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    story.append(sign_table)

    # Build PDF
    doc.build(story)
    return output_pdf_path

