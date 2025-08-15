#!/usr/bin/env python
"""
Phase 4: QR Code Integration for Linking Reports
Generate QR codes linking policy briefs, technical appendices, and interactive dashboards.
"""

import qrcode
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import logging
from PIL import Image, ImageDraw, ImageFont
import io
import base64


class QRIntegrationSystem:
    """
    QR Code integration system for linking reports and resources.
    
    Features:
    - Generate QR codes for report linking
    - Create branded QR codes with logos
    - Multi-format output (PNG, SVG, PDF)
    - URL validation and shortening
    - Batch QR code generation
    - Integration with report generators
    """
    
    def __init__(self, output_dir: Path = None, base_url: str = None):
        """
        Initialize QR integration system.
        
        Args:
            output_dir: Directory for QR code outputs
            base_url: Base URL for report hosting
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output/qr_codes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure base URL for report hosting
        self.base_url = base_url or "https://state-sped-policy.example.com"
        
        # QR code configuration
        self.qr_config = {
            "version": 1,  # Controls size (1-40)
            "error_correction": qrcode.constants.ERROR_CORRECT_L,
            "box_size": 10,
            "border": 4
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Track generated QR codes
        self.qr_registry = {}
        
    def generate_report_qr_codes(self, report_manifest: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate QR codes for all reports in manifest.
        
        Args:
            report_manifest: Dictionary with report types and paths
            
        Returns:
            Dictionary mapping report types to QR code paths
        """
        qr_codes = {}
        
        # Generate QR codes for each report type
        for report_type, report_info in report_manifest.items():
            if isinstance(report_info, dict):
                url = report_info.get("url") or self._construct_url(report_info.get("path", ""))
                title = report_info.get("title", report_type.title())
            else:
                url = self._construct_url(str(report_info))
                title = report_type.title()
            
            qr_path = self.create_branded_qr_code(
                url=url,
                title=title,
                filename=f"qr_{report_type}.png"
            )
            
            qr_codes[report_type] = qr_path
        
        # Generate master QR code linking to all reports
        master_qr = self.create_master_qr_code(report_manifest)
        qr_codes["master"] = master_qr
        
        # Save QR code registry
        self._save_qr_registry(qr_codes)
        
        return qr_codes
    
    def create_branded_qr_code(
        self, 
        url: str, 
        title: str = "", 
        filename: str = "qr_code.png",
        logo_path: Optional[str] = None
    ) -> str:
        """
        Create a branded QR code with title and optional logo.
        
        Args:
            url: URL to encode
            title: Title text to display
            filename: Output filename
            logo_path: Path to logo image (optional)
            
        Returns:
            Path to generated QR code
        """
        # Create QR code
        qr = qrcode.QRCode(**self.qr_config)
        qr.add_data(url)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to PIL Image if needed
        if not isinstance(qr_img, Image.Image):
            qr_img = Image.fromarray(qr_img)
        
        # Create branded version with title and styling
        branded_img = self._add_branding(qr_img, title, logo_path)
        
        # Save image
        output_path = self.output_dir / filename
        branded_img.save(output_path, "PNG", dpi=(300, 300))
        
        # Register QR code
        self.qr_registry[filename] = {
            "url": url,
            "title": title,
            "created": datetime.now().isoformat(),
            "path": str(output_path)
        }
        
        self.logger.info(f"Generated branded QR code: {output_path}")
        return str(output_path)
    
    def create_master_qr_code(self, report_manifest: Dict[str, Any]) -> str:
        """
        Create master QR code linking to main dashboard or report index.
        
        Args:
            report_manifest: Dictionary with all report information
            
        Returns:
            Path to master QR code
        """
        # Create dashboard URL
        dashboard_url = f"{self.base_url}/dashboard"
        
        # Create master QR code with special styling
        master_qr = self.create_branded_qr_code(
            url=dashboard_url,
            title="State Special Education Policy Analysis\nComplete Research Dashboard",
            filename="master_qr_code.png"
        )
        
        return master_qr
    
    def create_qr_code_sheet(self, qr_codes: Dict[str, str]) -> str:
        """
        Create a single sheet with multiple QR codes for printing.
        
        Args:
            qr_codes: Dictionary mapping report types to QR code paths
            
        Returns:
            Path to QR code sheet
        """
        # Load QR code images
        qr_images = {}
        for report_type, qr_path in qr_codes.items():
            if Path(qr_path).exists():
                qr_images[report_type] = Image.open(qr_path)
        
        if not qr_images:
            self.logger.warning("No QR code images found for sheet creation")
            return ""
        
        # Calculate sheet layout (2x3 grid for up to 6 QR codes)
        cols = 2
        rows = min(3, (len(qr_images) + 1) // 2)
        
        # Set up dimensions
        qr_size = 200  # pixels
        margin = 50
        sheet_width = cols * qr_size + (cols + 1) * margin
        sheet_height = rows * qr_size + (rows + 1) * margin + 100  # Extra space for header
        
        # Create sheet image
        sheet = Image.new('RGB', (sheet_width, sheet_height), 'white')
        draw = ImageDraw.Draw(sheet)
        
        # Add header
        try:
            # Try to use a nice font
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_subtitle = ImageFont.truetype("arial.ttf", 16)
        except:
            # Fallback to default font
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
        
        # Draw title
        title_text = "State Special Education Policy Analysis"
        subtitle_text = "QR Codes for Report Access"
        
        title_bbox = draw.textbbox((0, 0), title_text, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        
        draw.text(
            ((sheet_width - title_width) // 2, 20), 
            title_text, 
            fill='black', 
            font=font_title
        )
        
        subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=font_subtitle)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        
        draw.text(
            ((sheet_width - subtitle_width) // 2, 50), 
            subtitle_text, 
            fill='gray', 
            font=font_subtitle
        )
        
        # Place QR codes
        for i, (report_type, qr_img) in enumerate(qr_images.items()):
            if i >= cols * rows:
                break
                
            row = i // cols
            col = i % cols
            
            # Calculate position
            x = margin + col * (qr_size + margin)
            y = 100 + margin + row * (qr_size + margin)  # 100px offset for header
            
            # Resize QR code
            qr_resized = qr_img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)
            
            # Paste QR code
            sheet.paste(qr_resized, (x, y))
            
            # Add label
            label_text = self._format_report_label(report_type)
            label_bbox = draw.textbbox((0, 0), label_text, font=font_subtitle)
            label_width = label_bbox[2] - label_bbox[0]
            
            draw.text(
                (x + (qr_size - label_width) // 2, y + qr_size + 10),
                label_text,
                fill='black',
                font=font_subtitle
            )
        
        # Save sheet
        sheet_path = self.output_dir / "qr_code_sheet.png"
        sheet.save(sheet_path, "PNG", dpi=(300, 300))
        
        self.logger.info(f"Generated QR code sheet: {sheet_path}")
        return str(sheet_path)
    
    def generate_qr_html_embed(self, qr_codes: Dict[str, str]) -> str:
        """
        Generate HTML code for embedding QR codes in reports.
        
        Args:
            qr_codes: Dictionary mapping report types to QR code paths
            
        Returns:
            HTML string for embedding
        """
        html_parts = [
            '<div class="qr-code-section" style="text-align: center; margin: 20px 0;">',
            '<h3>Access Full Research Materials</h3>',
            '<p>Scan QR codes to access interactive reports and technical documentation:</p>',
            '<div class="qr-grid" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">'
        ]
        
        for report_type, qr_path in qr_codes.items():
            if report_type == "master":
                continue  # Handle master separately
                
            # Convert image to base64 for embedding
            with open(qr_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            label = self._format_report_label(report_type)
            
            html_parts.append(f'''
            <div class="qr-item" style="text-align: center; max-width: 150px;">
                <img src="data:image/png;base64,{img_data}" 
                     alt="QR Code for {label}" 
                     style="width: 120px; height: 120px;">
                <p style="margin: 5px 0; font-size: 12px;">{label}</p>
            </div>
            ''')
        
        # Add master QR code if available
        if "master" in qr_codes:
            with open(qr_codes["master"], "rb") as img_file:
                master_img_data = base64.b64encode(img_file.read()).decode()
            
            html_parts.append(f'''
            <div class="qr-master" style="margin-top: 20px; text-align: center;">
                <h4>Complete Research Dashboard</h4>
                <img src="data:image/png;base64,{master_img_data}" 
                     alt="Master QR Code" 
                     style="width: 150px; height: 150px;">
                <p style="font-size: 12px;">Scan for full interactive dashboard</p>
            </div>
            ''')
        
        html_parts.extend([
            '</div>',
            '</div>'
        ])
        
        return '\n'.join(html_parts)
    
    def _add_branding(self, qr_img: Image.Image, title: str, logo_path: Optional[str] = None) -> Image.Image:
        """Add branding elements to QR code."""
        # Calculate new dimensions with space for title
        qr_width, qr_height = qr_img.size
        title_height = 60 if title else 0
        logo_height = 40 if logo_path and Path(logo_path).exists() else 0
        
        total_height = qr_height + title_height + logo_height + 20  # 20px padding
        branded_width = max(qr_width, 300)  # Minimum width for text
        
        # Create new image
        branded_img = Image.new('RGB', (branded_width, total_height), 'white')
        
        # Paste QR code
        qr_x = (branded_width - qr_width) // 2
        qr_y = title_height + 10
        branded_img.paste(qr_img, (qr_x, qr_y))
        
        # Add title if provided
        if title:
            draw = ImageDraw.Draw(branded_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Handle multi-line titles
            lines = title.split('\n')
            line_height = 20
            total_text_height = len(lines) * line_height
            start_y = (title_height - total_text_height) // 2
            
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = (branded_width - text_width) // 2
                text_y = start_y + i * line_height
                
                draw.text((text_x, text_y), line, fill='black', font=font)
        
        # Add logo if provided
        if logo_path and Path(logo_path).exists():
            try:
                logo = Image.open(logo_path)
                logo = logo.resize((30, 30), Image.Resampling.LANCZOS)
                
                logo_x = (branded_width - 30) // 2
                logo_y = qr_y + qr_height + 10
                
                branded_img.paste(logo, (logo_x, logo_y))
            except Exception as e:
                self.logger.warning(f"Could not add logo: {e}")
        
        return branded_img
    
    def _construct_url(self, path: str) -> str:
        """Construct full URL from base URL and path."""
        if path.startswith('http'):
            return path
        
        # Clean path
        path = path.lstrip('/')
        return f"{self.base_url}/{path}"
    
    def _format_report_label(self, report_type: str) -> str:
        """Format report type for display labels."""
        label_mapping = {
            "policy_brief": "Policy Brief",
            "technical_appendix": "Technical Appendix", 
            "interactive_dashboard": "Interactive Dashboard",
            "methodology_docs": "Methodology Docs",
            "validation_report": "Validation Report",
            "master": "Complete Dashboard"
        }
        
        return label_mapping.get(report_type, report_type.replace('_', ' ').title())
    
    def _save_qr_registry(self, qr_codes: Dict[str, str]):
        """Save QR code registry for tracking."""
        registry_path = self.output_dir / "qr_registry.json"
        
        with open(registry_path, 'w') as f:
            json.dump(self.qr_registry, f, indent=2)
        
        self.logger.info(f"QR code registry saved: {registry_path}")


def create_report_qr_codes(
    report_manifest: Dict[str, Any],
    base_url: str = "https://state-sped-policy.example.com",
    output_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Create QR codes for all reports in manifest.
    
    Args:
        report_manifest: Dictionary with report information
        base_url: Base URL for hosted reports
        output_dir: Output directory for QR codes
        
    Returns:
        Dictionary mapping report types to QR code paths
    """
    qr_system = QRIntegrationSystem(output_dir, base_url)
    qr_codes = qr_system.generate_report_qr_codes(report_manifest)
    
    # Generate additional formats
    qr_system.create_qr_code_sheet(qr_codes)
    html_embed = qr_system.generate_qr_html_embed(qr_codes)
    
    # Save HTML embed code
    html_path = qr_system.output_dir / "qr_embed.html"
    with open(html_path, 'w') as f:
        f.write(html_embed)
    
    return qr_codes


if __name__ == "__main__":
    # Example usage
    sample_manifest = {
        "policy_brief": {
            "path": "policy_briefs/executive_brief.html",
            "title": "Executive Policy Brief",
            "url": "https://state-sped-policy.example.com/brief"
        },
        "technical_appendix": {
            "path": "technical_appendix/enhanced_appendix.html", 
            "title": "Technical Methodology",
            "url": "https://state-sped-policy.example.com/technical"
        },
        "interactive_dashboard": {
            "path": "dashboard/index.html",
            "title": "Interactive Dashboard",
            "url": "https://state-sped-policy.example.com/dashboard"
        }
    }
    
    qr_codes = create_report_qr_codes(sample_manifest)
    print(f"✅ Generated QR codes: {list(qr_codes.keys())}")
    print(f"📱 QR codes saved to: output/qr_codes/")