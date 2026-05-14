from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NoOpImageRunner(BaseImageRunner):
    """Stub runner for real-robot tasks where online rollouts aren't available."""

    def run(self, policy, **kwargs):
        return {"test_mean_score": 0.0}
