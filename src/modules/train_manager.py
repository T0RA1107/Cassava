class TrainManager:
    def __init__(
        self,
        train_loss_init=float('inf'),
        valid_loss_init=float('inf'),
        update_rate=0.9
    ):
        """_summary_

        Args:
            train_loss_init (float): initial loss
            valid_loss_init (float): initial loss
            update_rate (float, optional): Threshold of how improved loss for whether to update. Defaults to 0.9.
        """
        self.train_losses = []
        self.valid_losses = []
        self.train_loss_best = train_loss_init
        self.valid_loss_best = valid_loss_init
        self.update_rate = update_rate

    def check_and_save_weight(self, loss_train, loss_valid):
        self.train_losses.append(loss_train)
        self.valid_losses.append(loss_valid)
        if loss_train < self.train_loss_best * self.update_rate and loss_valid < self.valid_loss_best * self.update_rate:
            self.train_loss_best = loss_train
            self.valid_loss_best = loss_valid
            return True
        else:
            return False
