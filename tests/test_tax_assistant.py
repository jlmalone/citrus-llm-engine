"""
Tests for tax assistant
"""
import pytest


class TestTaxAssistant:
    """Test tax assistant"""

    @pytest.mark.asyncio
    async def test_office_supplies_deduction(self):
        """Test office supplies deduction"""
        from app.ml.tax_assistant import tax_assistant

        result = await tax_assistant.analyze_deduction(
            description="Office supplies from Staples",
            amount=50.0,
            category="Office Supplies",
            region="US",
            tax_year=2024
        )

        assert result["deduction_type"] == "fully_deductible"
        assert result["deductible_percentage"] == 100.0
        assert result["deductible_amount"] == 50.0

    @pytest.mark.asyncio
    async def test_meals_deduction(self):
        """Test meals deduction (50%)"""
        from app.ml.tax_assistant import tax_assistant

        result = await tax_assistant.analyze_deduction(
            description="Business lunch",
            amount=100.0,
            category="Meals",
            region="US",
            tax_year=2024
        )

        assert result["deduction_type"] == "partially_deductible"
        assert result["deductible_percentage"] == 50.0
        assert result["deductible_amount"] == 50.0

    @pytest.mark.asyncio
    async def test_vehicle_deduction(self):
        """Test vehicle deduction"""
        from app.ml.tax_assistant import tax_assistant

        result = await tax_assistant.analyze_deduction(
            description="Gas for business trip",
            amount=75.0,
            category="Vehicle",
            region="US",
            tax_year=2024
        )

        assert result["deduction_type"] in ["partially_deductible", "conditional"]
        assert len(result["documentation_required"]) > 0

    @pytest.mark.asyncio
    async def test_ca_tax_rules(self):
        """Test Canada tax rules"""
        from app.ml.tax_assistant import tax_assistant

        result = await tax_assistant.analyze_deduction(
            description="Office supplies",
            amount=50.0,
            category="Office Supplies",
            region="CA",
            tax_year=2024
        )

        assert result["cra_category"] is not None

    @pytest.mark.asyncio
    async def test_gb_tax_rules(self):
        """Test UK tax rules"""
        from app.ml.tax_assistant import tax_assistant

        result = await tax_assistant.analyze_deduction(
            description="Office supplies",
            amount=50.0,
            category="Office Supplies",
            region="GB",
            tax_year=2024
        )

        assert result["hmrc_category"] is not None
