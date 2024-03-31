
class LearningRateManager():
    def __init__(self, learning_rate_config):
        self.learning_rate = learning_rate_config["initial"]
        self.schedule_learning_rate = learning_rate_config["schedule_learning_rate"]
        self.learning_rate_divisor = learning_rate_config["learning_rate_divisor"]
        self.schedule_learning_rate_steps = learning_rate_config["schedule_learning_rate_steps"]

        self.steps_done = 0

    def get_learning_rate(self):
        if not self.schedule_learning_rate:
            return self.learning_rate
        self.steps_done += 1
        if self.steps_done >= self.schedule_learning_rate_steps:
            self.steps_done = 0
            self.learning_rate = self.learning_rate / self.learning_rate_divisor
        return self.learning_rate
