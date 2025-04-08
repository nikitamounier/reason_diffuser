import numpy as np
import matplotlib.pyplot as plt
from manim import *


class BackMasker(Scene):
    def construct(self):
        # Configuration
        num_blocks = 4  # Reduced to 4 blocks to fit on screen
        block_length = 32
        backmasking_frequency = 2  # Adjusted for fewer blocks
        backmasking_threshold = 0.4
        backmasking_alpha = 5.0
        backmasking_intensity = 0.5

        # Create title
        title = Text("LLM Backmasking Simulation", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Create blocks
        blocks = []
        block_scores = []
        token_confidences = []  # Track token-level confidence
        block_group = VGroup()
        mask_indicators = []  # To track which blocks are masked

        for i in range(num_blocks):
            # Create main block (initially all masked/black)
            block = Rectangle(
                height=1, width=3, fill_opacity=0.5, fill_color=BLACK
            )  # Made wider
            block.set_stroke(GRAY, 2)

            # Create mask overlay (initially fully visible)
            mask = Rectangle(
                height=1, width=3, fill_opacity=0.8, fill_color=BLACK, stroke_opacity=0
            )
            mask.move_to(block.get_center())

            if i > 0:
                block.next_to(blocks[-1], RIGHT, buff=0.5)  # Increased buffer
                mask.next_to(blocks[-1], RIGHT, buff=0.5)

            blocks.append(block)
            mask_indicators.append(mask)

            # Initial score (not shown since blocks start masked)
            score = 0.0  # Will be set during generation
            block_scores.append(score)

            # Simulate token confidences within each block
            token_confidences.append(np.random.uniform(0.3, 0.9, size=block_length))

            # Add score text (initially empty since blocks are masked)
            score_text = Text("", font_size=20)
            score_text.next_to(block, DOWN, buff=0.1)
            block.score_text = score_text

            # Group block, mask and score
            block_group.add(block, mask, score_text)

        # Center the blocks
        block_group.center()

        # Show initial blocks
        self.play(FadeIn(block_group))
        self.wait(1)

        # Legend for masked vs unmasked
        legend_group = VGroup()

        unmasked_sample = Rectangle(
            height=0.5, width=1, fill_opacity=0.5, fill_color=GRAY
        )
        unmasked_label = Text("Unmasked Block Group", font_size=16)
        unmasked_label.next_to(unmasked_sample, RIGHT, buff=0.3)

        masked_sample = Rectangle(
            height=0.5, width=1, fill_opacity=0.5, fill_color=BLACK
        )
        masked_label = Text("Masked Block Group", font_size=16)
        masked_label.next_to(masked_sample, RIGHT, buff=0.3)

        # Add partial masking example
        partial_sample = Rectangle(
            height=0.5, width=1, fill_opacity=0.5, fill_color=BLACK
        )
        # Create a gradient effect to show partial masking
        partial_overlay = Rectangle(
            height=0.5, width=0.6, fill_opacity=0.3, fill_color=GRAY, stroke_opacity=0
        )
        partial_overlay.align_to(partial_sample, LEFT)
        partial_group = VGroup(partial_sample, partial_overlay)
        partial_label = Text("Partially Masked (Token-Level)", font_size=16)
        partial_label.next_to(partial_group, RIGHT, buff=0.3)

        legend_group.add(
            unmasked_sample,
            unmasked_label,
            masked_sample,
            masked_label,
            partial_group,
            partial_label,
        )
        legend_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend_group.to_corner(DR, buff=0.5)

        self.play(FadeIn(legend_group))

        # Add explanation for gray shading
        shading_explanation = Text(
            "Gray shading represents how unmasked a block is", font_size=16
        )
        token_explanation = Text(
            "Backmasking selectively regenerates based on PRM score and token confidence",
            font_size=16,
        )
        explanations = VGroup(shading_explanation, token_explanation)
        explanations.arrange(DOWN, buff=0.2)
        explanations.next_to(legend_group, UP, buff=0.5)
        self.play(FadeIn(explanations))

        # Simulation loop
        current_position = 0
        while current_position < num_blocks:
            # Generate next block
            if current_position < num_blocks:
                # Simulate block generation
                self.play(
                    blocks[current_position].animate.set_fill(GREEN, opacity=0.7),
                    run_time=0.5,
                )

                # Set predefined scores for demonstration
                if current_position < 2:
                    # First 2 blocks have high scores
                    new_score = np.random.uniform(0.8, 0.95)
                elif current_position == 2:
                    # 3rd block has low score - exactly 0.4
                    new_score = 0.4
                else:
                    # Last block has varied score
                    new_score = np.random.uniform(0.6, 0.9)

                block_scores[current_position] = new_score
                new_score_text = Text(f"{new_score:.2f}", font_size=20)
                new_score_text.next_to(blocks[current_position], DOWN, buff=0.1)

                # Unmask based on score quality - higher score = more gray (less masked)
                # For a score of 0.4, it should be mostly black with a tinge of gray
                # For scores of 0.8-0.9, they should be mostly gray (mostly unmasked)
                gray_level = (
                    new_score * 0.6
                )  # Scale to make visual difference more apparent
                mask_opacity = 1.0 - new_score  # Higher score = less mask

                self.play(
                    Transform(blocks[current_position].score_text, new_score_text),
                    mask_indicators[current_position].animate.set_fill(
                        opacity=mask_opacity
                    ),
                    blocks[current_position].animate.set_fill(GRAY, opacity=gray_level),
                    run_time=0.5,
                )

                # If we just generated the 3rd block (index 2) with the 0.4 score,
                # immediately apply backmasking to all blocks (1-3)
                if current_position == 2:
                    # Create a text explanation
                    explanation = Text(
                        "Backmasking triggered by low quality block (0.4)",
                        font_size=24,
                    )
                    explanation.to_edge(DOWN, buff=1)
                    self.play(Write(explanation))

                    # Mask all blocks 0-2, but with PARTIAL masking based on token confidence
                    masked_blocks = [0, 1, 2]

                    # First, show token-level confidence visualization
                    token_conf_text = Text(
                        "Analyzing token-level confidence...", font_size=24
                    )
                    token_conf_text.next_to(explanation, UP, buff=0.5)
                    self.play(Write(token_conf_text))

                    # For each block, create a visualization of token confidence
                    token_visuals = []
                    for i in masked_blocks:
                        # Create mini-rectangles to represent tokens with varying confidence
                        token_group = VGroup()
                        for j in range(8):  # Show 8 representative tokens per block
                            # Use actual token confidences to determine opacity
                            conf = token_confidences[i][j * 4]  # Sample every 4th token
                            token_rect = Rectangle(
                                height=0.2,
                                width=0.3,
                                fill_opacity=0.8,
                                fill_color=RED if conf < 0.6 else GREEN,
                                stroke_width=1,
                            )
                            if j > 0:
                                token_rect.next_to(token_group[-1], RIGHT, buff=0.05)
                            token_group.add(token_rect)

                        token_group.scale(0.8)
                        token_group.next_to(blocks[i], UP, buff=0.2)
                        token_visuals.append(token_group)

                    # Show token confidence visualization
                    self.play(*[FadeIn(tv) for tv in token_visuals], run_time=1)
                    self.wait(1)

                    # Now apply partial masking based on PRM score and token confidence
                    for i in masked_blocks:
                        # Create partial masking effect - some parts more masked than others
                        # Higher PRM score = less overall masking
                        base_opacity = 0.9 - (block_scores[i] * 0.3)

                        # Create a partial mask with varying opacity
                        partial_mask = VGroup()
                        for j in range(6):  # Divide block into 6 sections
                            # Use token confidence to determine mask opacity for this section
                            section_conf = np.mean(
                                token_confidences[i][j * 5 : (j + 1) * 5]
                            )
                            section_opacity = base_opacity * (1.1 - section_conf)

                            section = Rectangle(
                                height=1,
                                width=0.5,
                                fill_opacity=section_opacity,
                                fill_color=BLACK,
                                stroke_opacity=0,
                            )

                            if j > 0:
                                section.next_to(partial_mask[-1], RIGHT, buff=0)
                            else:
                                section.align_to(blocks[i], LEFT)

                            partial_mask.add(section)

                        # Position the partial mask over the block
                        partial_mask.move_to(blocks[i].get_center())

                        # Animate the transition to partial masking
                        self.play(
                            FadeOut(mask_indicators[i]),
                            FadeIn(partial_mask),
                            blocks[i].animate.set_fill(GRAY, opacity=0.3),
                            run_time=0.8,
                        )

                        # Replace the old mask indicator with the partial mask
                        mask_indicators[i] = partial_mask

                        # Remove score text when masked
                        empty_text = Text("", font_size=20)
                        empty_text.next_to(blocks[i], DOWN, buff=0.1)
                        self.play(
                            Transform(blocks[i].score_text, empty_text),
                            run_time=0.2,
                        )

                    # Clean up token visualizations
                    self.play(
                        *[FadeOut(tv) for tv in token_visuals], FadeOut(token_conf_text)
                    )

                    # Regeneration phase
                    unmask_text = Text(
                        "Partially regenerating blocks based on token confidence...",
                        font_size=24,
                    )
                    unmask_text.next_to(explanation, UP, buff=0.5)
                    self.play(Write(unmask_text))

                    # Highlight all masked blocks simultaneously
                    highlights = []
                    for i in masked_blocks:
                        highlight = SurroundingRectangle(blocks[i], color=YELLOW)
                        highlights.append(highlight)

                    self.play(*[Create(h) for h in highlights], run_time=0.5)
                    self.wait(0.5)

                    # Regenerate with higher scores, but maintain partial regeneration visualization
                    for i in masked_blocks:
                        # Significantly improve all scores
                        if i == 0:
                            new_score = 0.92  # First block gets very high score
                        elif i == 1:
                            new_score = 0.88  # Second block gets high score
                        else:  # i == 2
                            new_score = 0.85  # Third block gets much better score

                        block_scores[i] = new_score
                        new_score_text = Text(f"{new_score:.2f}", font_size=20)
                        new_score_text.next_to(blocks[i], DOWN, buff=0.1)

                        # Update visualization - show partial regeneration
                        # Some sections become more visible than others based on token confidence
                        animations = []

                        # Update the partial mask sections with new opacities
                        for j, section in enumerate(mask_indicators[i]):
                            section_conf = np.mean(
                                token_confidences[i][j * 5 : (j + 1) * 5]
                            )
                            # Higher confidence and higher score = less masking
                            new_opacity = (1.0 - new_score) * (1.1 - section_conf)
                            animations.append(
                                section.animate.set_fill(opacity=new_opacity)
                            )

                        # Update block color and score
                        animations.extend(
                            [
                                blocks[i].animate.set_fill(
                                    GRAY, opacity=new_score * 0.6
                                ),
                                Transform(blocks[i].score_text, new_score_text),
                            ]
                        )

                        self.play(*animations, run_time=1.0)

                    self.play(*[FadeOut(h) for h in highlights], run_time=0.3)
                    self.play(FadeOut(unmask_text))
                    self.wait(1)

                    # Add explanation about partial regeneration
                    partial_regen_text = Text(
                        "Only low-confidence tokens were regenerated", font_size=24
                    )
                    partial_regen_text.next_to(explanation, UP, buff=0.5)
                    self.play(Write(partial_regen_text))
                    self.wait(1.5)
                    self.play(FadeOut(partial_regen_text), FadeOut(explanation))

                current_position += 1

            # No need for additional backmasking since we already did it after block 3
            if current_position == num_blocks:
                # Final state
                final_text = Text("Generation Complete", font_size=30)
                final_text.to_edge(DOWN, buff=1)
                self.play(Write(final_text))

                # Show average score
                avg_score = np.mean(block_scores)
                avg_text = Text(f"Average Score: {avg_score:.2f}", font_size=24)
                avg_text.next_to(final_text, UP, buff=0.5)
                self.play(Write(avg_text))

                # Show comparison with and without backmasking
                comparison_text = Text(
                    "Partial regeneration improved quality while preserving good content",
                    font_size=24,
                )
                comparison_text.next_to(avg_text, UP, buff=0.5)
                self.play(Write(comparison_text))

        self.wait(2)
        self.play(FadeOut(final_text), FadeOut(avg_text), FadeOut(comparison_text))

        # Run time comparison
        runtime_title = Text("Efficiency of Partial Regeneration", font_size=30)
        runtime_title.to_edge(UP, buff=1)
        self.play(Write(runtime_title))

        # Create a simple bar chart comparing full vs partial regeneration
        bar_labels = ["Full Regeneration", "Partial Regeneration"]

    def calculate_backmasking_probs(
        self, block_scores, backmasking_alpha=5.0, min_prob=0.01
    ):
        """
        Calculate backmasking probabilities for each block based on PRM scores.
        We use an exponential function to create a steeper curve that emphasizes poor scores more.
        """
        if not block_scores:
            return []

        # Convert to numpy array for easier manipulation
        scores = np.array(block_scores)

        # Apply exponential transformation
        probs = np.exp(-backmasking_alpha * scores)

        # Ensure minimum probability
        probs = np.maximum(probs, min_prob)

        # Normalize to [min_prob, 1]
        probs = min_prob + (1 - min_prob) * (probs - probs.min()) / (
            probs.max() - probs.min() + 1e-8
        )

        return probs


class BackmaskingExplanation(Scene):
    def construct(self):
        # Title
        title = Text("How Backmasking Works", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Create a simple diagram
        # 1. Initial generation
        step1 = Text("1. Generate blocks of text", font_size=24)
        step1.shift(UP * 2)

        # 2. Score evaluation
        step2 = Text("2. Evaluate quality with PRM model", font_size=24)
        step2.next_to(step1, DOWN, buff=0.5)

        # 3. Backmasking
        step3 = Text("3. Identify low-quality blocks", font_size=24)
        step3.next_to(step2, DOWN, buff=0.5)

        # 4. Regeneration
        step4 = Text("4. Selectively mask and regenerate", font_size=24)
        step4.next_to(step3, DOWN, buff=0.5)

        # 5. Final output
        step5 = Text("5. Produce higher quality output", font_size=24)
        step5.next_to(step4, DOWN, buff=0.5)

        # Show steps sequentially
        for step in [step1, step2, step3, step4, step5]:
            self.play(Write(step))
            self.wait(0.5)

        # Show formula for backmasking probability
        formula = MathTex(r"P(mask) = e^{-\alpha \cdot score}")
        formula.next_to(step5, DOWN, buff=1)
        formula_explanation = Text(
            "Higher α = more aggressive backmasking", font_size=20
        )
        formula_explanation.next_to(formula, DOWN, buff=0.3)

        self.play(Write(formula))
        self.play(Write(formula_explanation))

        self.wait(2)
