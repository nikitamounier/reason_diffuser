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

        legend_group.add(unmasked_sample, unmasked_label, masked_sample, masked_label)
        legend_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend_group.to_corner(DR, buff=0.5)

        self.play(FadeIn(legend_group))

        # Add explanation for gray shading
        shading_explanation = Text(
            "Gray shading represents how unmasked a block is", font_size=16
        )
        shading_explanation.next_to(legend_group, UP, buff=0.5)
        self.play(FadeIn(shading_explanation))

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
                    # 3rd block has low score
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

                current_position += 1

            # Apply backmasking if we've seen a low quality block (even if current one is good)
            should_backmask = False
            if current_position > 0:
                # Check if any previous block has a low score
                if any(
                    score <= backmasking_threshold
                    for score in block_scores[:current_position]
                ):
                    should_backmask = True

            if should_backmask or current_position == num_blocks:
                # Calculate backmasking probabilities
                backmasking_probs = self.calculate_backmasking_probs(
                    block_scores[:current_position], backmasking_alpha
                )

                # Create a text explanation
                explanation = Text(
                    f"Backmasking after position {current_position} due to low quality block",
                    font_size=24,
                )
                explanation.to_edge(DOWN, buff=1)
                self.play(Write(explanation))

                # First pass: Identify blocks to mask based on scores
                masked_blocks = []
                for i in range(current_position):
                    mask_probability = backmasking_probs[i]

                    # Always mask the low-quality block (position 2 with score 0.4)
                    # Also mask some previous blocks even if they had good scores
                    if block_scores[i] < backmasking_threshold or (
                        np.random.random() < mask_probability
                        and i < current_position - 1
                    ):
                        # Calculate mask opacity based on how bad the score is
                        mask_opacity = 0.9 * (1 - block_scores[i])

                        # Visualize masking by showing black overlay
                        self.play(
                            mask_indicators[i].animate.set_fill(opacity=mask_opacity),
                            blocks[i].animate.set_fill(
                                BLACK, opacity=0.2
                            ),  # Very little gray for masked blocks
                            run_time=0.3,
                        )

                        # Remove score text when masked
                        empty_text = Text("", font_size=20)
                        empty_text.next_to(blocks[i], DOWN, buff=0.1)
                        self.play(
                            Transform(blocks[i].score_text, empty_text), run_time=0.2
                        )

                        masked_blocks.append(i)

                # Second pass: Parallel unmasking and resampling
                if masked_blocks:
                    unmask_text = Text("Resampling masked blocks...", font_size=24)
                    unmask_text.next_to(explanation, UP, buff=0.5)
                    self.play(Write(unmask_text))

                    # Highlight all masked blocks simultaneously
                    highlights = []
                    for i in masked_blocks:
                        highlight = SurroundingRectangle(blocks[i], color=YELLOW)
                        highlights.append(highlight)

                    self.play(*[Create(h) for h in highlights], run_time=0.5)
                    self.wait(0.5)

                    # Gradually unmask all blocks in parallel
                    for step in range(3):  # Show iterative improvement
                        animations = []

                        for i in masked_blocks:
                            # Partially unmask
                            new_opacity = mask_indicators[i].get_fill_opacity() * 0.6

                            # Improve score gradually - higher improvement in second pass
                            if i == 2:  # The problematic block gets a big improvement
                                new_score = (
                                    0.4 + (step + 1) * 0.15
                                )  # Gradually improve to ~0.85
                            else:  # Other blocks get smaller improvements
                                improvement = (0.95 - block_scores[i]) * (1 - step / 3)
                                new_score = block_scores[i] + improvement

                            block_scores[i] = new_score

                            # Only show score text in final step
                            if step == 2:
                                new_score_text = Text(f"{new_score:.2f}", font_size=20)
                            else:
                                new_score_text = Text("", font_size=20)

                            new_score_text.next_to(blocks[i], DOWN, buff=0.1)

                            # Update visualization - gradually turn from black to gray
                            # Higher score = more gray (more unmasked)
                            gray_level = (
                                new_score * 0.6
                            )  # Scale to make visual difference more apparent

                            animations.extend(
                                [
                                    mask_indicators[i].animate.set_fill(
                                        opacity=1.0 - new_score
                                    ),
                                    blocks[i].animate.set_fill(
                                        GRAY, opacity=gray_level
                                    ),
                                    Transform(blocks[i].score_text, new_score_text),
                                ]
                            )

                        self.play(*animations, run_time=0.8)

                    self.play(*[FadeOut(h) for h in highlights], run_time=0.3)
                    self.play(FadeOut(unmask_text))

                self.play(FadeOut(explanation))

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
        comparison_text = Text("Backmasking improves average quality", font_size=24)
        comparison_text.next_to(avg_text, UP, buff=0.5)
        self.play(Write(comparison_text))

        self.wait(2)

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
