from .models.metrics import MetricsCalculation
from .models.training import train

NUM_EPOCHS = 50



def main():
    train(
        TRAIN_HISTORY_STORAGE,
        MODEL_NAME, model, NUM_EPOCHS,
        train_loader, loc_criterion_cls, conf_pos_criterion_cls, conf_neg_criterion_cls, optimizer, valid_loader,
        alpha=10,
        lr_scheduler=lr_scheduler, lr_scheduler_1st_epoch=3, use_valid_loss_for_scheduler=False, init_epoch_callback=init_epoch_callback, max_grad_norm=MAX_GRAD_NORM,
        calc_mIoU=MetricsCalculation.valid, calc_accuracy=MetricsCalculation.valid,
        calc_gt_boxes_detected=MetricsCalculation.both, calc_mean_num_pos_def_boxes=MetricsCalculation.both,
        report_priors_scales_usage=MetricsCalculation.both,
        prediction_example_choosing='first', show_only_last_plot=False,
        # early_stopping=True, es_patience=2, es_rel_threshold=5e-2,
        # early_stopping=True, es_patience=3, es_rel_threshold=5e-2,
        # early_stopping=True, es_patience=5, es_rel_threshold=5e-2,
        # early_stopping=True, es_patience=7, es_rel_threshold=5e-2,
        # early_stopping=True, es_patience=9, es_rel_threshold=1e-2,
        # early_stopping=False, es_patience=19, es_rel_threshold=1e-2, use_valid_loss_for_es=True,
        early_stopping=False, es_patience=19, es_rel_threshold=1e-2, use_valid_loss_for_es=False,
        device=DEVICE)


if __name__ == "__main__":
    raise NotImplementedError
    main()