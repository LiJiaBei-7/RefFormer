from transformers import AdamW, get_linear_schedule_with_warmup


def get_param(args, model):
    lr = args.lr
    no_decay = ['bias', 'LayerNorm.weight']
    params_weight = []
    params_no_weight = []

    for name, param in model.named_parameters():
        if not any(nd in name for nd in no_decay):
            params_weight.append(param)
        else:
            params_no_weight.append(param)

    optimizer_grouped_parameters = [
        {'params': params_weight,
         'weight_decay': 1e-2
         },
        {'params': params_no_weight,
         'weight_decay': 0.0
         }
    ]

    return optimizer_grouped_parameters


def build_optimizer(args, model):
    param = get_param(args, model)
    optimizer = AdamW(param, lr=args.lr, eps=1e-8)
    return optimizer

def build_schedule(args, optimizer):
    # epochs = 50
    # training steps 的数量: [number of batches] x [number of epochs].
    # total_steps = steps_num * epochs
    total_steps = args['epochs'] * args['step_per_epoch']
    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(total_steps * args['num_warmup_steps'])

    # 设计 learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return scheduler
