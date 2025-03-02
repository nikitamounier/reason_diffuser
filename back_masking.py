import numpy as np
import matplotlib.pyplot as plt
from manim import *


class BackmaskingSimulation(Scene):
    def construct(self):
        # Configuration
        num_blocks = 15
        block_length = 32
        backmasking_frequency = 5
        backmasking_threshold = 0.4
        backmasking_alpha = 5.0
        backmasking_intensity = 0.5

        # Create title
        title = Text("LLM Backmasking Simulation", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Add explanation about shading
        shading_explanation = Text(
            "Block shading indicates generation progress", font_size=20
        )
        shading_explanation.next_to(title, DOWN, buff=0.3)
        self.play(Write(shading_explanation))

        # Create blocks
        blocks = []
        block_scores = []
        block_group = VGroup()

        for i in range(num_blocks):
            block = Rectangle(height=1, width=2, fill_opacity=0.5, fill_color=BLUE)
            block.set_stroke(WHITE, 2)
            if i > 0:
                block.next_to(blocks[-1], RIGHT, buff=0.2)
            blocks.append(block)
            # Random initial score
            score = np.random.uniform(0.2, 0.8)
            block_scores.append(score)

            # Add score text
            score_text = Text(f"{score:.2f}", font_size=20)
            score_text.next_to(block, DOWN, buff=0.1)
            block.score_text = score_text

            # Group block and score
            block_group.add(block, score_text)

        # Center the blocks
        block_group.center()

        # Show initial blocks
        self.play(FadeIn(block_group))
        self.wait(1)

        # Fade out the shading explanation after blocks are shown
        self.play(FadeOut(shading_explanation))

        # Simulation loop
        for step in range(num_blocks // backmasking_frequency):
            backmasking_step = (step + 1) * backmasking_frequency

            # Generate blocks up to the backmasking step
            if step > 0:  # Skip first iteration as blocks are already created
                for i in range(
                    backmasking_step - backmasking_frequency, backmasking_step
                ):
                    if i >= num_blocks:
                        break

                    # Simulate block generation
                    self.play(
                        blocks[i].animate.set_fill(GREEN, opacity=0.7), run_time=0.5
                    )

                    # Update score
                    new_score = np.random.uniform(0.2, 0.8)
                    block_scores[i] = new_score
                    new_score_text = Text(f"{new_score:.2f}", font_size=20)
                    new_score_text.next_to(blocks[i], DOWN, buff=0.1)

                    self.play(
                        Transform(blocks[i].score_text, new_score_text), run_time=0.5
                    )

                    # Immediately check if we need to regenerate this block
                    if new_score < backmasking_threshold:
                        # Visualize backmasking
                        self.play(
                            blocks[i].animate.set_fill(RED, opacity=0.7), run_time=0.3
                        )

                        # Simulate regeneration
                        self.wait(0.5)
                        regenerated_score = np.random.uniform(
                            0.4, 0.9
                        )  # Better score after regeneration
                        block_scores[i] = regenerated_score
                        regenerated_score_text = Text(
                            f"{regenerated_score:.2f}", font_size=20
                        )
                        regenerated_score_text.next_to(blocks[i], DOWN, buff=0.1)

                        self.play(
                            blocks[i].animate.set_fill(BLUE, opacity=0.5),
                            Transform(blocks[i].score_text, regenerated_score_text),
                            run_time=0.5,
                        )

            # Check if we need to apply backmasking
            if backmasking_step <= num_blocks:
                # Calculate backmasking probabilities
                backmasking_probs = self.calculate_backmasking_probs(
                    block_scores[:backmasking_step], backmasking_alpha
                )

                # Create a text explanation
                explanation = Text(
                    f"Backmasking after block {backmasking_step}", font_size=24
                )
                explanation.to_edge(DOWN, buff=1)
                self.play(Write(explanation))

                # Highlight blocks that will be backmasked
                for i in range(backmasking_step):
                    # Apply backmasking based on probability and threshold
                    if (
                        block_scores[i] < backmasking_threshold
                        or backmasking_probs[i] > 0.5
                    ):
                        # Visualize backmasking
                        self.play(
                            blocks[i].animate.set_fill(RED, opacity=0.7), run_time=0.3
                        )

                        # Simulate regeneration
                        self.wait(0.5)
                        new_score = np.random.uniform(
                            0.4, 0.9
                        )  # Better score after regeneration
                        block_scores[i] = new_score
                        new_score_text = Text(f"{new_score:.2f}", font_size=20)
                        new_score_text.next_to(blocks[i], DOWN, buff=0.1)

                        self.play(
                            blocks[i].animate.set_fill(BLUE, opacity=0.5),
                            Transform(blocks[i].score_text, new_score_text),
                            run_time=0.5,
                        )

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
