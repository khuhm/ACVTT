from data_handler.DataHandler import DataHandler
from data_writer.DataWriter import DataWriter
from models.Model import Model
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, args=None):
        # info.
        self.do_val = args.do_val
        self.do_test = args.do_test
        self.save_freq = args.save_freq
        self.val_freq = args.val_freq
        self.test_freq = args.test_freq
        self.show_train_data = args.show_train_data
        self.test_only = args.test_only
        self.test_split = args.test_split
        self.plot_metric = args.plot_metric

        # epoch
        self.start_epoch = 0
        self.last_epoch = args.last_epoch

        # initialize data handler
        self.data_handler = DataHandler(args)

        # initialize model
        self.model = Model(args)

        # load model if resume or test
        if args.load_model:
            self.start_epoch = self.model.load(args.model_name)

        # initialize writer
        self.data_writer = DataWriter(args)

    def run_training(self):

        # for epochs
        progress_epoch = tqdm(range(self.start_epoch, self.last_epoch + 1), desc='epoch iter', leave=False)
        for epoch in progress_epoch:
            # init train setting
            self.model.init_train_mode()
            # train loops
            progress_bar = tqdm(getattr(self.data_handler, 'train_data_loader'), desc='train iter', leave=False)
            for i, data in enumerate(progress_bar):
                self.model.one_train_loop(data)
                if self.show_train_data:
                    self.data_handler.show_data(data)
                pass

            # get losses
            loss = self.model.get_avg_loss()

            # plot losses
            self.data_writer.plot('train', loss, epoch)

            # save models
            if epoch % self.save_freq == 0:
                self.model.save(epoch)

            # validation
            if self.do_val and (epoch % self.val_freq == 0):
                self.model.init_val_mode()
                progress_bar = tqdm(getattr(self.data_handler, 'val_data_loader'), desc='vai iter', leave=False)
                for i, data in enumerate(progress_bar):
                    self.model.one_val_loop(data)
                    pass

                # get losses
                loss = self.model.get_avg_loss()
                # plot losses
                self.data_writer.plot('val', loss, epoch)

            # test if available
            if self.do_test and (epoch > 0) and (epoch % self.test_freq == 0):
                self.run_testing(epoch)

    def run_testing(self, epoch=None):
        if epoch is None:
            curr_epoch = self.start_epoch
        else:
            curr_epoch = epoch
        # test
        self.model.init_test_mode()
        progress_bar = tqdm(getattr(self.data_handler, 'test_data_loader'), desc='test iter', leave=False)
        for i, data in enumerate(progress_bar):
            self.model.one_test_loop(data, 'test')

        # save metrics
        self.data_writer.save_metrics([self.model.get_metrics(), self.model.get_avg_metrics()], curr_epoch)

        # calculate metrics
        metrics = self.model.get_avg_metrics()

        if self.test_only:
            print(curr_epoch, metrics)

        # plot metrics
        if self.plot_metric:
            self.data_writer.plot('test', metrics, curr_epoch)
        pass

    def run_test_all(self, epoch=None):
        if epoch is None:
            curr_epoch = self.start_epoch
        else:
            curr_epoch = epoch
        # test
        self.model.init_test_mode()

        if 'train' in self.test_split:
            progress_bar = tqdm(getattr(self.data_handler, 'train_data_loader'), desc='train iter', leave=False)
            for i, data in enumerate(progress_bar):
                self.model.one_test_loop(data, 'train')

        if 'val' in self.test_split:
            progress_bar = tqdm(getattr(self.data_handler, 'val_data_loader'), desc='val iter', leave=False)
            for i, data in enumerate(progress_bar):
                self.model.one_test_loop(data, 'val')

        if 'test' in self.test_split:
            progress_bar = tqdm(getattr(self.data_handler, 'test_data_loader'), desc='test iter', leave=False)
            for i, data in enumerate(progress_bar):
                self.model.one_test_loop(data, 'test')
        pass
