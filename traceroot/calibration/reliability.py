"""Reliability diagrams: ASCII terminal and SVG/HTML visual representations."""

from typing import List, Optional
from traceroot.calibration.metrics import CalibrationBin, CalibrationSummary


class ReliabilityDiagram:
    """Generates visual and terminal representations of model confidence calibration."""

    @staticmethod
    def render_ascii(summary: CalibrationSummary) -> str:
        """Render a formatted ASCII reliability table and diagram for CLI output."""
        lines = []
        lines.append("=" * 68)
        lines.append("                     RELIABILITY DIAGRAM                      ")
        lines.append("=" * 68)
        lines.append(f" ECE: {summary.expected_calibration_error:.4f} | MCE: {summary.maximum_calibration_error:.4f} | Brier: {summary.brier_score:.4f}")
        lines.append("-" * 68)
        lines.append(f"{'Bin Range':<15} | {'Count':<6} | {'Mean Conf':<10} | {'Accuracy':<10} | {'Gap':<8} | {'Diagram'}")
        lines.append("-" * 68)

        for b in summary.bins:
            bin_str = f"[{b.lower_bound:.2f}, {b.upper_bound:.2f}]"
            count_str = str(b.count)
            conf_str = f"{b.mean_confidence:.3f}" if b.count > 0 else "-"
            acc_str = f"{b.accuracy:.3f}" if b.count > 0 else "-"
            gap_str = f"{b.calibration_gap:.3f}" if b.count > 0 else "-"

            # Visual bar comparison: Conf (#) vs Acc (*)
            bar_len = 10
            conf_pos = int(round(b.mean_confidence * bar_len)) if b.count > 0 else 0
            acc_pos = int(round(b.accuracy * bar_len)) if b.count > 0 else 0

            bar = [" "] * (bar_len + 1)
            if b.count > 0:
                bar[conf_pos] = "C"
                bar[acc_pos] = "A" if acc_pos != conf_pos else "X"
            diagram_str = f"[{''.join(bar)}] (C=Conf, A=Acc)"

            lines.append(f"{bin_str:<15} | {count_str:<6} | {conf_str:<10} | {acc_str:<10} | {gap_str:<8} | {diagram_str}")

        lines.append("=" * 68)
        return "\n".join(lines)

    @staticmethod
    def render_svg(summary: CalibrationSummary, width: int = 500, height: int = 400) -> str:
        """Generate a standalone SVG plot of the reliability curve."""
        margin = 50
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#1e1e2e" rx="8"/>',
            f'  <text x="{width//2}" y="30" fill="#cdd6f4" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Reliability Curve (ECE: {summary.expected_calibration_error:.3f})</text>',
            # Axes
            f'  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#6c7086" stroke-width="2"/>',
            f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#6c7086" stroke-width="2"/>',
            # Ideal diagonal (y = x)
            f'  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{margin}" stroke="#a6adc8" stroke-dasharray="4,4" stroke-width="1.5"/>',
            # Labels
            f'  <text x="{width//2}" y="{height - 15}" fill="#a6adc8" font-family="sans-serif" font-size="12" text-anchor="middle">Mean Predicted Confidence</text>',
            f'  <text x="15" y="{height//2}" fill="#a6adc8" font-family="sans-serif" font-size="12" text-anchor="middle" transform="rotate(-90 15 {height//2})">Empirical Accuracy</text>',
        ]

        # Plot bins
        for b in summary.bins:
            if b.count > 0:
                cx = margin + int(b.mean_confidence * plot_w)
                cy = (height - margin) - int(b.accuracy * plot_h)
                # Calibration bar / point
                svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="5" fill="#89b4fa" stroke="#b4befe" stroke-width="1.5"/>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
