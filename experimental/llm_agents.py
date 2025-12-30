#!/usr/bin/env python3
## Experimental: LLM-Powered Explanations

"""Modular agentic flow for portfolio analysis explanations."""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add api directory to path for imports
api_dir = Path(__file__).parent.parent / 'api'
sys.path.insert(0, str(api_dir))

from .llm_service import generate_llm_explanation, is_llm_available
from schemas import Portfolio, PortfolioAnalysisResponse

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Base class for modular agents in the explanation flow."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            context: Context dictionary with portfolio, analysis, recommendations, etc.
        
        Returns:
            Dictionary with agent's output
        """
        pass
    
    def format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM prompt."""
        return str(context)


class RecommendationInterpreterAgent(Agent):
    """Agent that interprets portfolio recommendations using LLM."""
    
    def __init__(self):
        super().__init__(
            name="recommendation_interpreter",
            description="Interprets portfolio recommendations in natural language"
        )
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate natural language interpretation of recommendations."""
        if not is_llm_available():
            return {
                "success": False,
                "error": "LLM service not available",
                "explanation": None
            }
        
        recommendations = context.get("recommendations", [])
        risk_level = context.get("risk_level", "unknown")
        portfolio_metrics = context.get("portfolio_metrics", {})
        feature_conclusions = context.get("feature_conclusions", [])
        security_scores = context.get("security_scores", [])
        
        prompt = self._build_prompt(recommendations, risk_level, portfolio_metrics, feature_conclusions, security_scores)
        
        # Debug: Log the LLM request
        logger.info(f"[{self.name}] LLM Request Details:")
        logger.info(f"  Context keys: {list(context.keys())}")
        logger.info(f"  Recommendations count: {len(recommendations)}")
        logger.info(f"  Feature conclusions count: {len(feature_conclusions)}")
        logger.info(f"  Security scores count: {len(security_scores)}")
        if feature_conclusions:
            logger.info(f"  Feature conclusion categories: {[c.get('category') for c in feature_conclusions]}")
        logger.info(f"  Prompt length: {len(prompt)} characters")
        logger.debug(f"[{self.name}] Full LLM Prompt:\n{prompt}")
        
        try:
            explanation = generate_llm_explanation(prompt)
            return {
                "success": True,
                "explanation": explanation,
                "agent": self.name
            }
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanation": None
            }
    
    def _build_prompt(
        self, 
        recommendations: List[str], 
        risk_level: str, 
        metrics: Dict[str, Any], 
        feature_conclusions: List[Dict] = None,
        security_scores: List[Dict] = None
    ) -> str:
        """Build prompt for recommendation interpretation."""
        prompt = f"""You are a financial advisor analyzing portfolio recommendations for an individual investor. 
Write in a clear, conversational tone that an average investor can understand. Avoid excessive jargon.

Portfolio Risk Level: {risk_level.upper()}

Portfolio Metrics:
- Total Value: ${metrics.get('total_value', 0):,.2f}
- Number of Positions: {metrics.get('n_positions', 0)}
- Maximum Position Weight: {metrics.get('max_position_weight', 0)*100:.1f}%
- Top 3 Concentration: {metrics.get('top_3_concentration', 0)*100:.1f}%
- Concentration HHI: {metrics.get('concentration_hhi', 0):.3f}

"""
        
        # Add per-security analysis if available
        if security_scores:
            prompt += self._format_security_scores(security_scores, metrics)
        
        # Add z-score context if available
        if metrics.get('ae_weighted_z_score') is not None:
            prompt += self._format_zscore_context(metrics)
        
        # Add feature conclusions if available
        if feature_conclusions:
            prompt += "Technical Feature Analysis:\n"
            for i, conclusion in enumerate(feature_conclusions, 1):
                prompt += f"{i}. [{conclusion.get('category', 'Unknown')}] {conclusion.get('severity', 'unknown').upper()} - {conclusion.get('finding', '')}\n"
                prompt += f"   Implication: {conclusion.get('implication', '')}\n"
            prompt += "\n"
        
        prompt += "Recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        
        prompt += """
Please provide a clear, actionable explanation for the investor. Include:

1. **Overall Assessment**: A 2-3 sentence summary of their portfolio's health
2. **Key Concerns**: What specific issues need attention (reference specific holdings if data available)
3. **Recommended Actions**: What they should consider doing, in priority order
4. **Context**: How their portfolio compares to typical market behavior (if z-score data available)

Keep the explanation professional but accessible. Use specific numbers and holding names when available.
Do not use excessive bullet points - write in natural paragraphs where appropriate."""
        
        return prompt
    
    def _format_security_scores(self, security_scores: List[Dict], metrics: Dict) -> str:
        """Format per-security scores for the prompt."""
        prompt = "Individual Holding Analysis:\n"
        prompt += "-" * 40 + "\n"
        
        # Separate into categories, distinguishing material from immaterial
        anomalous_material = [s for s in security_scores if s.get('consensus_anomaly') and s.get('is_material', True)]
        anomalous_immaterial = [s for s in security_scores if s.get('consensus_anomaly') and not s.get('is_material', True)]
        warnings = [s for s in security_scores if s.get('status') == 'warning' and not s.get('consensus_anomaly')]
        normal = [s for s in security_scores if s.get('status') == 'normal']
        
        # Collect sectors for diversification analysis
        sectors = {}
        for s in security_scores:
            sector = s.get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(s.get('symbol'))
        
        if anomalous_material:
            prompt += "\n🚨 ANOMALOUS HOLDINGS - PRIORITY ATTENTION:\n"
            for s in anomalous_material:
                prompt += self._format_single_security(s)
        
        if anomalous_immaterial:
            prompt += "\n⚠️ ANOMALOUS BUT IMMATERIAL (< 1% of portfolio):\n"
            for s in anomalous_immaterial:
                prompt += self._format_single_security(s, brief=True)
            prompt += "   (These are flagged but too small to significantly impact portfolio risk)\n"
        
        if warnings:
            prompt += "\n⚡ HOLDINGS TO MONITOR:\n"
            for s in warnings:
                prompt += self._format_single_security(s)
        
        if normal:
            prompt += "\n✓ NORMAL HOLDINGS:\n"
            for s in normal[:3]:  # Limit to top 3 normal holdings
                prompt += self._format_single_security(s, brief=True)
            if len(normal) > 3:
                prompt += f"   ... and {len(normal) - 3} more normal holdings\n"
        
        # Sector summary
        if len(sectors) > 1:
            prompt += f"\nSector Exposure: {len(sectors)} sectors\n"
            for sector, symbols in sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True):
                prompt += f"   • {sector}: {', '.join(symbols)}\n"
        elif len(sectors) == 1:
            sector = list(sectors.keys())[0]
            prompt += f"\n⚠️ Single Sector Exposure: All holdings in {sector}\n"
        
        # Anomaly summary with materiality context
        n_total = len(security_scores)
        n_anomalous = len(anomalous_material) + len(anomalous_immaterial)
        n_material_anomalous = len(anomalous_material)
        
        prompt += f"\nSummary: {n_anomalous}/{n_total} holdings flagged as anomalous"
        if anomalous_immaterial:
            prompt += f" ({n_material_anomalous} material, {len(anomalous_immaterial)} immaterial)"
        prompt += "\n\n"
        
        return prompt
    
    def _format_single_security(self, s: Dict, brief: bool = False) -> str:
        """Format a single security for the prompt."""
        symbol = s.get('symbol', 'Unknown')
        weight = s.get('weight', 0) * 100
        value = s.get('position_value', 0)
        sector = s.get('sector', 'Unknown')
        
        line = f"   • {symbol} ({sector}): {weight:.1f}% of portfolio (${value:,.0f})"
        
        # Add gain/loss info if available
        gain_loss_pct = s.get('gain_loss_percent')
        gain_loss_dollars = s.get('gain_loss_dollars')
        if gain_loss_pct is not None:
            if gain_loss_pct >= 0:
                line += f" [+{gain_loss_pct:.0f}% gain, +${gain_loss_dollars:,.0f}]"
            else:
                line += f" [{gain_loss_pct:.0f}% loss, -${abs(gain_loss_dollars):,.0f}]"
        
        line += "\n"
        
        if not brief:
            # Use human-readable z-score display
            z_display = s.get('ae_z_score_display')
            if z_display:
                line += f"     Z-score: {z_display}\n"
            
            reason = s.get('status_reason')
            if reason:
                line += f"     Status: {reason}\n"
            
            # Add materiality note for large positions
            mat_note = s.get('materiality_note')
            if mat_note and s.get('weight', 0) > 0.25:
                line += f"     Note: {mat_note}\n"
        
        return line
    
    def _format_zscore_context(self, metrics: Dict) -> str:
        """Format z-score context for the prompt."""
        ae_z = metrics.get('ae_weighted_z_score')
        if_z = metrics.get('if_weighted_z_score')
        vs_market = metrics.get('portfolio_vs_market')
        
        prompt = "Portfolio vs Market Context:\n"
        
        if ae_z is not None:
            prompt += f"- Portfolio z-score (AE): {ae_z:+.2f}\n"
        if if_z is not None:
            prompt += f"- Portfolio z-score (IF): {if_z:+.2f}\n"
        
        if vs_market:
            interpretations = {
                "significantly_more_anomalous": "This portfolio is behaving SIGNIFICANTLY differently from the typical market (>2 standard deviations). This warrants close attention.",
                "more_anomalous_than_typical": "This portfolio shows more unusual patterns than typical (1-2 standard deviations above normal).",
                "calmer_than_typical": "This portfolio is actually calmer and more stable than typical market behavior.",
                "similar_to_market": "This portfolio is behaving similarly to typical market patterns."
            }
            prompt += f"- Interpretation: {interpretations.get(vs_market, vs_market)}\n"
        
        prompt += "\n"
        return prompt


class PortfolioContextAgent(Agent):
    """Agent that provides portfolio context-aware explanations."""
    
    def __init__(self):
        super().__init__(
            name="portfolio_context",
            description="Provides explanations with full portfolio context"
        )
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation with full portfolio context."""
        if not is_llm_available():
            return {
                "success": False,
                "error": "LLM service not available",
                "explanation": None
            }
        
        portfolio = context.get("portfolio")
        analysis = context.get("analysis")
        recommendations = context.get("recommendations", [])
        feature_conclusions = context.get("feature_conclusions", [])
        security_scores = context.get("security_scores", [])
        basket_stats = context.get("basket_stats")
        
        prompt = self._build_prompt(portfolio, analysis, recommendations, feature_conclusions, security_scores, basket_stats)
        
        # Debug: Log the LLM request
        logger.info(f"[{self.name}] LLM Request Details:")
        logger.info(f"  Context keys: {list(context.keys())}")
        logger.info(f"  Portfolio positions: {len(portfolio.positions) if portfolio else 0}")
        logger.info(f"  Recommendations count: {len(recommendations)}")
        logger.info(f"  Feature conclusions count: {len(feature_conclusions)}")
        logger.info(f"  Security scores count: {len(security_scores)}")
        if feature_conclusions:
            logger.info(f"  Feature conclusion categories: {[c.get('category') for c in feature_conclusions]}")
        logger.info(f"  Prompt length: {len(prompt)} characters")
        logger.debug(f"[{self.name}] Full LLM Prompt:\n{prompt}")
        
        try:
            explanation = generate_llm_explanation(prompt, max_tokens=3000)
            return {
                "success": True,
                "explanation": explanation,
                "agent": self.name
            }
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanation": None
            }
    
    def _build_prompt(
        self, 
        portfolio: Optional[Portfolio], 
        analysis: Optional[Dict], 
        recommendations: List[str], 
        feature_conclusions: List[Dict] = None,
        security_scores: List[Dict] = None,
        basket_stats: Dict = None
    ) -> str:
        """Build prompt with full portfolio context."""
        prompt = """You are a financial advisor providing comprehensive portfolio analysis to an individual investor.
Write in a clear, conversational tone. The investor has a basic understanding of investing but is not a professional.
Be specific about their actual holdings and provide actionable guidance.

"""
        
        # Portfolio holdings with cost basis
        if portfolio:
            prompt += "YOUR PORTFOLIO HOLDINGS:\n"
            prompt += "=" * 50 + "\n"
            for symbol, position in portfolio.positions.items():
                prompt += f"• {symbol}: {position.shares} shares"
                if position.cost_basis:
                    prompt += f" (purchased at ${position.cost_basis:.2f}/share)"
                prompt += "\n"
            prompt += f"\nBenchmark: {portfolio.benchmark}\n\n"
        
        # Analysis results
        if analysis:
            prompt += "ANALYSIS RESULTS:\n"
            prompt += "=" * 50 + "\n"
            prompt += f"• Overall Risk Level: {analysis.get('risk_level', 'unknown').upper()}\n"
            prompt += f"• Concentration Risk: {analysis.get('concentration_risk', 'unknown')}\n"
            prompt += f"• Volatility Risk: {analysis.get('volatility_risk', 'unknown')}\n"
            
            model_results = analysis.get('model_results', {})
            if model_results:
                ae = model_results.get('autoencoder', {})
                prompt += f"• Anomaly Score: {ae.get('score', 0):.4f} (threshold: {ae.get('threshold', 0):.4f})\n"
                prompt += f"• Anomaly Detected: {'YES ⚠️' if ae.get('is_anomaly', False) else 'No ✓'}\n"
            
            metrics = analysis.get('portfolio_metrics', {})
            if metrics:
                prompt += f"• Total Portfolio Value: ${metrics.get('total_value', 0):,.2f}\n"
                prompt += f"• Number of Positions: {metrics.get('n_positions', 0)}\n"
                prompt += f"• Largest Position: {metrics.get('max_position_weight', 0)*100:.1f}% of portfolio\n"
                prompt += f"• Top 3 Holdings: {metrics.get('top_3_concentration', 0)*100:.1f}% of portfolio\n"
            prompt += "\n"
        
        # Per-security detailed analysis
        if security_scores:
            prompt += "DETAILED HOLDING ANALYSIS:\n"
            prompt += "=" * 50 + "\n"
            
            # Categorize securities with materiality consideration
            anomalous_material = [s for s in security_scores if s.get('consensus_anomaly') and s.get('is_material', True)]
            anomalous_immaterial = [s for s in security_scores if s.get('consensus_anomaly') and not s.get('is_material', True)]
            warnings = [s for s in security_scores if s.get('status') == 'warning' and not s.get('consensus_anomaly')]
            normal = [s for s in security_scores if s.get('status') == 'normal']
            
            # Sector analysis
            sectors = {}
            for s in security_scores:
                sector = s.get('sector', 'Unknown')
                weight = s.get('weight', 0)
                if sector not in sectors:
                    sectors[sector] = {'symbols': [], 'weight': 0}
                sectors[sector]['symbols'].append(s.get('symbol'))
                sectors[sector]['weight'] += weight
            
            if anomalous_material:
                prompt += "\n🚨 HOLDINGS REQUIRING IMMEDIATE ATTENTION:\n"
                for s in anomalous_material:
                    prompt += self._format_detailed_security(s)
            
            if anomalous_immaterial:
                prompt += "\n⚠️ ANOMALOUS BUT LOW IMPACT (< 1% of portfolio):\n"
                for s in anomalous_immaterial:
                    prompt += self._format_detailed_security(s, brief=True)
                prompt += "  Note: These positions are flagged but too small to significantly affect portfolio risk.\n"
            
            if warnings:
                prompt += "\n⚡ HOLDINGS TO MONITOR:\n"
                for s in warnings:
                    prompt += self._format_detailed_security(s)
            
            if normal:
                prompt += "\n✓ HEALTHY HOLDINGS:\n"
                for s in normal:
                    prompt += self._format_detailed_security(s, brief=True)
            
            # Sector diversification summary
            prompt += "\nSECTOR ALLOCATION:\n"
            for sector, data in sorted(sectors.items(), key=lambda x: x[1]['weight'], reverse=True):
                weight_pct = data['weight'] * 100
                symbols = ', '.join(data['symbols'])
                if weight_pct > 50:
                    prompt += f"  🔴 {sector}: {weight_pct:.1f}% ({symbols}) - OVER-CONCENTRATED\n"
                elif weight_pct > 30:
                    prompt += f"  🟠 {sector}: {weight_pct:.1f}% ({symbols}) - high allocation\n"
                else:
                    prompt += f"  ✓ {sector}: {weight_pct:.1f}% ({symbols})\n"
            
            if len(sectors) <= 2:
                prompt += "\n⚠️ LOW SECTOR DIVERSIFICATION: Portfolio concentrated in only {len(sectors)} sector(s)\n"
            
            # Overall stats
            n_material_anomalous = len(anomalous_material)
            n_total_anomalous = len(anomalous_material) + len(anomalous_immaterial)
            prompt += f"\nPortfolio Health Summary:\n"
            prompt += f"  • {n_total_anomalous}/{len(security_scores)} holdings flagged as anomalous"
            if anomalous_immaterial:
                prompt += f" ({n_material_anomalous} material, {len(anomalous_immaterial)} immaterial)"
            prompt += f"\n  • {len(warnings)} holdings with warnings, {len(normal)} healthy\n\n"
        
        # Z-score context
        if analysis and analysis.get('portfolio_metrics', {}).get('ae_weighted_z_score') is not None:
            metrics = analysis.get('portfolio_metrics', {})
            ae_z = metrics.get('ae_weighted_z_score')
            vs_market = metrics.get('portfolio_vs_market')
            
            prompt += "MARKET COMPARISON:\n"
            prompt += "=" * 50 + "\n"
            prompt += f"Your portfolio's weighted z-score: {ae_z:+.2f}\n"
            
            if vs_market == "significantly_more_anomalous":
                prompt += "⚠️ Your portfolio is behaving VERY differently from typical market patterns.\n"
                prompt += "This is more than 2 standard deviations from normal - unusual behavior warranting review.\n"
            elif vs_market == "more_anomalous_than_typical":
                prompt += "⚡ Your portfolio shows somewhat unusual patterns compared to typical market behavior.\n"
            elif vs_market == "calmer_than_typical":
                prompt += "✓ Your portfolio is actually more stable than typical market conditions.\n"
            else:
                prompt += "Your portfolio is behaving similarly to typical market patterns.\n"
            prompt += "\n"
        
        # Basket statistics context
        if basket_stats:
            prompt += f"(Based on analysis of {basket_stats.get('n_securities', 0)} securities in the market universe)\n\n"
        
        # Feature conclusions
        if feature_conclusions:
            prompt += "TECHNICAL INDICATORS:\n"
            prompt += "=" * 50 + "\n"
            for conclusion in feature_conclusions:
                severity = conclusion.get('severity', 'unknown').upper()
                icon = "🚨" if severity == "HIGH" else "⚡" if severity == "MEDIUM" else "ℹ️"
                prompt += f"{icon} [{conclusion.get('category', 'Unknown')}] {conclusion.get('finding', '')}\n"
                prompt += f"   → {conclusion.get('implication', '')}\n"
            prompt += "\n"
        
        # Recommendations
        prompt += "SYSTEM RECOMMENDATIONS:\n"
        prompt += "=" * 50 + "\n"
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        
        prompt += """

INSTRUCTIONS FOR YOUR RESPONSE:
Please provide a comprehensive yet accessible explanation that:

1. **Executive Summary** (2-3 sentences): What's the bottom line on this portfolio's health?

2. **What's Working**: Any positive aspects of the portfolio

3. **Key Concerns**: 
   - Which specific holdings are problematic and why
   - Use the actual ticker symbols and percentages
   - Explain what the anomaly detection is telling us in plain language

4. **Recommended Actions**:
   - Prioritize: what should they do first, second, third?
   - Be specific: "Consider reducing your NVDA position from 35% to 15%" not just "diversify"
   - Explain the reasoning behind each recommendation

5. **Important Context**:
   - How this compares to typical portfolios
   - Any caveats or limitations of the analysis

Write as if speaking directly to the investor. Be honest about concerns but not alarmist.
Avoid excessive jargon - if you use a technical term, briefly explain it."""
        
        return prompt
    
    def _format_detailed_security(self, s: Dict, brief: bool = False) -> str:
        """Format detailed security information."""
        symbol = s.get('symbol', 'Unknown')
        weight = s.get('weight', 0) * 100
        value = s.get('position_value', 0)
        shares = s.get('shares', 0)
        price = s.get('current_price', 0)
        sector = s.get('sector', 'Unknown')
        is_material = s.get('is_material', True)
        
        line = f"\n{symbol} ({sector}):\n"
        line += f"  Position: {shares:.0f} shares @ ${price:.2f} = ${value:,.0f} ({weight:.1f}% of portfolio)\n"
        
        # Add cost basis and gain/loss if available
        cost_basis = s.get('cost_basis')
        gain_loss_pct = s.get('gain_loss_percent')
        gain_loss_dollars = s.get('gain_loss_dollars')
        
        if cost_basis is not None:
            line += f"  Cost Basis: ${cost_basis:.2f}/share"
            if gain_loss_pct is not None:
                if gain_loss_pct >= 0:
                    line += f" → Unrealized Gain: +${gain_loss_dollars:,.0f} (+{gain_loss_pct:.1f}%)"
                else:
                    line += f" → Unrealized Loss: -${abs(gain_loss_dollars):,.0f} ({gain_loss_pct:.1f}%)"
            line += "\n"
        
        if not brief:
            ae_score = s.get('ae_score', 0)
            if_score = s.get('if_score', 0)
            ae_z_display = s.get('ae_z_score_display')
            if_z_display = s.get('if_z_score_display')
            
            line += f"  Model Scores: AE={ae_score:.4f}, IF={if_score:.4f}"
            if ae_z_display:
                line += f"\n  Risk vs Market: AE {ae_z_display}"
                if if_z_display:
                    line += f", IF {if_z_display}"
            line += "\n"
            
            reason = s.get('status_reason')
            if reason:
                line += f"  Assessment: {reason}\n"
            
            # Add materiality context
            if not is_material:
                line += f"  ⚪ Note: Position too small ({weight:.2f}%) to significantly impact portfolio\n"
            elif weight > 0.40:
                line += f"  🔴 CRITICAL: Position represents {weight:.1f}% - severe concentration risk\n"
            elif weight > 0.25:
                line += f"  🟠 WARNING: Large position ({weight:.1f}%) - monitor closely\n"
        
        return line


class AgenticFlow:
    """Orchestrates multiple agents in a flow."""
    
    def __init__(self, agents: Optional[List[Agent]] = None):
        """
        Initialize agentic flow.
        
        Args:
            agents: List of agents to execute (default: recommendation interpreter)
        """
        if agents is None:
            agents = [RecommendationInterpreterAgent()]
        self.agents = agents
        self.execution_history = []
    
    def add_agent(self, agent: Agent):
        """Add an agent to the flow."""
        self.agents.append(agent)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all agents in sequence.
        
        Args:
            context: Context dictionary with portfolio, analysis, recommendations, etc.
        
        Returns:
            Dictionary with results from all agents
        """
        results = {
            "flow_success": True,
            "agents_executed": [],
            "explanations": [],
            "errors": []
        }
        
        for agent in self.agents:
            logger.info(f"Executing agent: {agent.name}")
            try:
                result = agent.execute(context)
                results["agents_executed"].append(agent.name)
                
                if result.get("success"):
                    results["explanations"].append({
                        "agent": agent.name,
                        "explanation": result.get("explanation")
                    })
                else:
                    results["errors"].append({
                        "agent": agent.name,
                        "error": result.get("error")
                    })
                    if not is_llm_available():
                        results["flow_success"] = False
                
            except Exception as e:
                logger.error(f"Error executing agent {agent.name}: {e}")
                results["errors"].append({
                    "agent": agent.name,
                    "error": str(e)
                })
                results["flow_success"] = False
        
        self.execution_history.append(results)
        return results
    
    def get_combined_explanation(self, results: Dict[str, Any]) -> str:
        """Combine explanations from all agents into a single text."""
        explanations = results.get("explanations", [])
        if not explanations:
            return "No explanations generated."
        
        combined = []
        for exp in explanations:
            combined.append(f"## {exp['agent'].replace('_', ' ').title()}\n\n{exp['explanation']}\n")
        
        return "\n".join(combined)