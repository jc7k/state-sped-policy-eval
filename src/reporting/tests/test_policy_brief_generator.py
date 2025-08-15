"""
Tests for Policy Brief Generator

Tests the policy brief generation for non-technical audiences.
"""

import tempfile
from pathlib import Path

import pytest

from src.reporting.policy_brief_generator import PolicyBriefGenerator


class TestPolicyBriefGenerator:
    """Test PolicyBriefGenerator functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def generator(self, temp_output_dir):
        """PolicyBriefGenerator instance."""
        return PolicyBriefGenerator(output_dir=temp_output_dir)

    @pytest.fixture
    def mock_results(self):
        """Mock analysis results."""
        return {
            "causal": {"math_grade4_gap": {"coefficient": 0.15, "p_value": 0.06}},
            "robustness": {"bootstrap": {"significant": False}},
            "power": {"average_power": 0.65},
        }

    def test_init_creates_output_directory(self, temp_output_dir):
        """Test initialization creates output directory."""
        output_dir = temp_output_dir / "briefs"
        PolicyBriefGenerator(output_dir=output_dir)
        assert output_dir.exists()

    def test_create_policy_brief(self, generator, mock_results, temp_output_dir):
        """Test policy brief generation."""
        output_path = generator.create_policy_brief(mock_results, "test_brief.tex")

        expected_path = str(temp_output_dir / "test_brief.tex")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_policy_brief_content(self, generator, mock_results):
        """Test policy brief content structure."""
        output_path = generator.create_policy_brief(mock_results, "content_test.tex")

        with open(output_path) as f:
            content = f.read()

        # Check for key sections
        assert "Special Education Policy Reform" in content
        assert "Executive Summary" in content
        assert "Key Findings" in content
        assert "Policy Implications" in content
        assert "Recommendations" in content
        assert "Methodology at a Glance" in content

    def test_extract_key_findings(self, generator, mock_results):
        """Test key findings extraction."""
        findings = generator._extract_key_findings(mock_results)

        assert isinstance(findings, list)
        assert len(findings) > 0
        assert all(isinstance(finding, str) for finding in findings)

    def test_generate_implications(self, generator, mock_results):
        """Test implications generation."""
        implications = generator._generate_implications(mock_results)

        assert isinstance(implications, list)
        assert len(implications) > 0
        assert all(isinstance(imp, str) for imp in implications)

    def test_generate_recommendations(self, generator, mock_results):
        """Test recommendations generation."""
        recommendations = generator._generate_recommendations(mock_results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_create_infographic_data(self, generator, mock_results):
        """Test infographic data creation."""
        infographic_data = generator.create_infographic_data(mock_results)

        assert isinstance(infographic_data, dict)
        assert "title" in infographic_data
        assert "key_stat_1" in infographic_data
        assert "main_finding" in infographic_data
        assert "recommendation" in infographic_data

    def test_latex_formatting_validity(self, generator, mock_results):
        """Test that generated LaTeX has valid structure."""
        output_path = generator.create_policy_brief(mock_results, "latex_test.tex")

        with open(output_path) as f:
            content = f.read()

        # Check for proper LaTeX structure
        assert content.count("\\begin{document}") == 1
        assert content.count("\\end{document}") == 1
        assert "\\documentclass" in content
        assert "\\begin{itemize}" in content
        assert "\\end{itemize}" in content
        assert "\\begin{enumerate}" in content
        assert "\\end{enumerate}" in content

    def test_empty_results_handling(self, generator):
        """Test handling of empty results."""
        empty_results = {}

        # Should not raise an error
        output_path = generator.create_policy_brief(empty_results, "empty_test.tex")

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        # Should still have basic structure
        assert "\\begin{document}" in content
        assert "\\end{document}" in content

    def test_custom_filename_handling(self, generator, mock_results, temp_output_dir):
        """Test custom filename handling."""
        custom_filename = "custom_policy_brief_2024.tex"
        output_path = generator.create_policy_brief(mock_results, custom_filename)

        expected_path = str(temp_output_dir / custom_filename)
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_contact_information_inclusion(self, generator, mock_results):
        """Test that contact information is included."""
        output_path = generator.create_policy_brief(mock_results, "contact_test.tex")

        with open(output_path) as f:
            content = f.read()

        # Check for contact information
        assert "Jeff Chen" in content
        assert "jeffreyc1@alumni.cmu.edu" in content
        assert "Claude Code" in content

    def test_methodology_section_completeness(self, generator, mock_results):
        """Test that methodology section is complete and accessible."""
        output_path = generator.create_policy_brief(mock_results, "methodology_test.tex")

        with open(output_path) as f:
            content = f.read()

        # Check for methodology elements in plain language
        assert "50 states" in content
        assert "14 years" in content or "2009-2022" in content
        assert "difference-in-differences" in content or "staggered treatment" in content
        assert "robustness" in content
