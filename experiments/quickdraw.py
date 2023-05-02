import os
import torch

import clutils
from clutils import monitors
from clutils.experiments import Trainer
from clutils.experiments.utils import get_device, create_result_folder, create_optimizers
from clutils.metrics import accuracy
from clutils.strategies import PackedRehearsal
from clutils.models.RNN import SketchLSTM
from mylib.mlflow_utils import MlflowLogger
from mylib.lstm import MaskedSketchLSTM, create_lstm_masks, MaskedLSTMWeightRatioLogger
from mylib.aco import AntGraph, PheromoneStatLogger

def quickdraw_exp(args, is_aco=False):
    """
    required attributes of args:

    exp_name
    run_name
    method

    cpus_per_job

    mlruns_dir
    result_folder
    dataroot
    cuda: bool
    num_workers: int

    # Task INFO
    exp_type
    n_tasks: int
    classes_per_task: int
    output_size: int
    input_size: int
    pixel_in_input: int # Used to produce permutation and image sequence shape.
    max_label_value: int
    tasks_list: list[list[int]] # only for split mnist

    # Model Hyperparameters
    epochs: int
    hidden_size_rnn: int
    layers_rnn: int
    orthogonal: bool # Use orthogonal recurrent matrixes
    expand_output: int #Expand output layer dynamically if > 0

    # OPTIMIZER
    weight_decay: float
    learning_rate: float
    batch_size: int
    clip_grad: float
    use_sgd: bool

    # EXTRAS
    not_test: bool
    not_intermediate_test: bool
    monitor: bool # monitor metrics
    save: bool # save models

    # Replay Memory
    rehe_patterns: int # Number of rehearsal patterns per class in the memory. This controls all type of rehearsal.
    patterns_per_class_per_batch: int # can be -1 (disable this option, not rehearsal), 0 (add a minibatch equally split among classes) or >0 (specify patterns per class per minibatch)
    
    # ACO Parameters
    rehe_patterns_val: int
    patterns_per_class_per_batch_val: int
    aco_init_value: float
    num_ant: int
    aco_evaporation_rate: float
    aco_reward_coef: float
    aco_update_rule: int (0, 1, 2)
    """

    classes = ['hot dog', 'palm tree', 'moon', 'envelope', 'dumbbell', 'microwave',
               'onion', 'nail', 'paper clip', 'soccer ball',
               'drill', 'telephone',
               'airplane', 'dishwasher', 'chair', 'grass', 'rhinoceros', 'octopus',
               'cloud', 'bicycle', 'swan', 'picture frame', 'shorts', 'flying saucer',
               'basketball', 'harp', 'beard', 'binoculars', 'tiger', 'book', 'scissors',
               'raccoon', 'peanut', 'wheel', 'trombone', 'diamond', 'parachute', 'tractor',
               'windmill', 'alarm clock', 'clarinet', 'spider', 'violin', 'clock', 'tent',
               'belt', 'map', 'toe', 'bread', 'kangaroo', 'camera', 'duck', 'lipstick',
               'snowman', 'pickup truck', 'radio', 'truck', 'train', 'teapot', 'tree',
               'hourglass', 'eyeglasses', 'church', 'submarine', 'couch', 'umbrella',
               'whale', 'cooler', 'sword', 'table', 'skull', 'house', 'blackberry', 'bush', 'giraffe',
               'rainbow', 'yoga', 'mailbox', 'wristwatch', 'The Eiffel Tower', 'syringe', 'bulldozer',
               'door', 'zebra', 'beach', 'crown', 'screwdriver', 'bear', 'sink', 'teddy-bear', 'square',
               'cruise ship', 'waterslide', 'elbow', 'stereo', 'sweater', 'bandage', 'bat', 'The Mona Lisa',
               'sea turtle', 'butterfly', 'mosquito', 'tennis racquet', 'tornado', 'computer', 'bridge',
               'toothbrush', 'baseball bat', 'bench', 'finger', 'canoe', 'baseball', 'circle', 'banana',
               'bathtub', 'axe', 'lantern', 'birthday cake', 'castle', 'wine bottle', 'ant',
               'The Great Wall of China', 'bee', 'apple', 'arm', 'asparagus', 'angel', 'cup', 'carrot',
               'bucket', 'animal migration', 'cell phone', 'van', 'dolphin', 'bowtie', 'campfire', 'ceiling fan',
               'boomerang', 'aircraft carrier', 'cactus', 'cake', 'anvil', 'toothpaste', 'swing set', 'feather',
               'flashlight', 'garden hose', 'camel', 'calculator', 'diving board', 'chandelier', 'helmet',
               'squirrel', 'ambulance', 'bottlecap', 'hospital', 'coffee cup', 'watermelon', 'dresser', 'bed',
               'bird', 'cookie', 'underwear', 'drums', 'cat', 'calendar', 'bracelet', 'eraser', 'dog', 'barn',
               'spoon', 'sun', 'toilet', 'backpack', 'trumpet', 'frying pan', 'blueberry', 'toaster', 'floor lamp',
               'crocodile', 'police car', 'cow', 'basket', 'cello', 'golf club', 'school bus', 'hockey puck', 'fence',
               'donut', 'goatee', 'traffic light', 'hamburger', 'ear', 'compass', 'broccoli', 'skyscraper', 'fan',
               'hot air balloon', 'mountain', 'fork', 'face', 'crab', 'ice cream', 'foot', 'hat', 'candle', 'flip flops',
               'hammer', 'guitar', 'brain', 'stove', 'headphones', 'flamingo', 'flower', 'bus', 'hot tub', 'elephant',
               'fire hydrant', 'laptop', 'leaf', 'potato', 'hockey stick', 'lighter', 'hexagon', 'garden', 'marker',
               'keyboard', 'camouflage', 'knee', 'sheep', 'microphone', 'mushroom', 'light bulb', 'hand', 'key',
               'house plant', 'eye', 'matches', 'broom', 'knife', 'crayon', 'ocean', 'dragon', 'leg', 'horse',
               'zigzag', 'car', 'grapes', 'lightning', 'moustache', 'mouth', 'vase', 'fish', 'string bean',
               'lighthouse', 'ladder', 'television', 'helicopter', 'pillow', 'pencil', 'rollerskates', 'jail', 'mermaid',
               'jacket', 'megaphone', 'nose', 'pants', 'octagon', 'pizza', 'passport', 'pool', 'motorbike', 'snake',
               'pond', 'frog', 'pig', 'penguin', 'cannon', 'parrot', 'lobster', 'saw', 'strawberry', 'firetruck',
               'speedboat', 'popsicle', 'hurricane', 'see saw', 'saxophone', 'mug', 'piano', 'mouse', 'power outlet',
               'hedgehog', 'oven', 'shoe', 'rifle', 'roller coaster', 'peas', 'lion', 'rake', 'postcard', 'sock',
               'purse', 'sleeping bag', 'skateboard', 'necklace', 'stairs', 'lollipop', 'snowflake', 'rabbit', 'owl',
               'shovel', 'pear', 'remote control', 'star', 'scorpion', 'washing machine', 'monkey', 'pineapple', 'sandwich',
               'shark', 'sailboat', 'steak', 'stethoscope', 'wine glass', 'smiley face', 'streetlight', 'squiggle',
               'stop sign', 'line', 'pliers', 'paint can', 'panda', 'paintbrush', 't-shirt', 'fireplace', 'river',
               'snorkel', 'rain', 'triangle', 'suitcase', 'stitches', 'tooth', 'snail', 'spreadsheet']

    torch.set_num_threads(args.cpus_per_job)
    print(f'torch num_threads: {torch.get_num_threads()}')

    classes = classes[:100]
    classes = classes[10:]
    classes = [ [classes[i+j] for j in range(args.classes_per_task)] for i in range(0,len(classes),args.classes_per_task)]

    device = get_device(args.cuda)
    result_folder = create_result_folder(os.path.join(args.result_folder, args.run_name))

    result_logger = monitors.log.create_logger(result_folder, log_file='training_results.csv')
    metric_monitor = monitors.log.LogMetric(args.n_tasks, result_folder, 'acc', intermediate_results_file='intermediate_results.csv')
    monitors.log.write_configuration(args, result_folder)

    mlflowLogger = MlflowLogger(args.exp_name, mlruns_dir=args.mlruns_dir)
    mlflowLogger.start_run(args.run_name)
    mlflowLogger.log_params(vars(args))

    if is_aco:
        pheromone_stat_logger = PheromoneStatLogger((args.layers_rnn * 2 + 1), os.path.join(result_folder, 'pheromone_stat'))
        weight_ratio_logger_train = MaskedLSTMWeightRatioLogger(args.layers_rnn, result_folder, 'weight_ratios_train.csv')
        weight_ratio_logger_val = MaskedLSTMWeightRatioLogger(args.layers_rnn, result_folder, 'weight_ratios_val.csv')

    if is_aco:
        args.models = 'aco_lstm'
        model = MaskedSketchLSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
                        num_layers=args.layers_rnn, orthogonal=args.orthogonal)
        ant_graph_shape = [args.input_size] + [args.hidden_size_rnn] * (args.layers_rnn * 2) + [args.output_size]
        aco = AntGraph(ant_graph_shape, args.aco_init_value)
    else:
        args.models = 'lstm'
        model = SketchLSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
                    num_layers=args.layers_rnn, orthogonal=args.orthogonal)
    optimizer = create_optimizers({args.models: model}, args.learning_rate, args.weight_decay, args.use_sgd)[args.models]
    criterion = torch.nn.CrossEntropyLoss()

    model.output_type = clutils.OUTPUT_TYPE.LAST_OUT

    trainer = Trainer(model, optimizer, criterion, device, accuracy, clip_grad=args.clip_grad)

    if args.monitor:
        writer = monitors.plots.create_writer(result_folder)

    rehe = PackedRehearsal(args.rehe_patterns, args.patterns_per_class_per_batch)
    if is_aco:
        rehe_val = PackedRehearsal(args.rehe_patterns_val, args.patterns_per_class_per_batch_val)

    if args.save:
        clutils.experiments.utils.save_model(model, args.models, result_folder, version='_init')
        if is_aco:
            aco.save(os.path.join(result_folder, 'pheromone'), 'pheromone_init')

    dataset = clutils.datasets.CLQuickDraw(os.path.expanduser(args.dataroot), train_batch_size=args.batch_size,
            test_batch_size=500,  len_task_vector=0,
            task_vector_at_test=False)

    monitor_last_epoch = True # plot weights and gradients always or only for last epoch of each task

    result_logger.warning("model,task_id,epoch,train_acc,validation_acc,train_loss,validation_loss")

    for task_id in range(args.n_tasks):

        loader_task_train, loader_task_test = dataset.get_task_loaders(classes=classes[task_id])

        ######## VALIDATION BEFORE TRAINING ########
        for x,y in loader_task_test:
            x,y = x.to(device), y.to(device)

            validation_loss, validation_accuracy = trainer.test(x,y, task_id=None)
            metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

        metric_monitor.update_metrics(args.models, 'val', task_id, num_batches=len(loader_task_test), reset_averages=True)
        val_acc = metric_monitor.get_metric(args.models, 'val', 'acc', task_id)
        val_loss = metric_monitor.get_metric(args.models, 'val', 'loss', task_id)
        result_logger.warning(
            f"{args.models},"
            f"{task_id},0,"
            f"-1,"
            f"{val_acc},"
            f"-1,"
            f"{val_loss}"
        )
        mlflowLogger.log_metric('val_acc_before_task', val_acc, step=task_id)
        mlflowLogger.log_metric('val_loss_before_task', val_loss, step=task_id)

        ######## START SPECIFIC TASK ########
        
        if args.rehe_patterns > 0:
            loader_train = rehe.augment_dataset(loader_task_train)
        else:
            loader_train = loader_task_train

        if is_aco:
            best_masks_so_far = None
            best_aco_val_loss = float('inf')

        for epoch in range(1, args.epochs+1):
            result_logger.info(f"Task {task_id} - Epoch {epoch}/{args.epochs}")

            ######## TRAINING ########

            if is_aco:
                ant_path = aco.choose_path(args.num_ant)
                masks = aco.to_mask(ant_path)
                tensor_masks = create_lstm_masks(masks)
                model.set_masks(tensor_masks)
                weight_ratio_logger_train.log(model.get_weight_ratios())

            for id_batch, (x,y) in enumerate(loader_train):
                
                if args.rehe_patterns > 0:
                    x, y = rehe.concat_to_batch(x,y)

                x,y = x.to(device), y.to(device)

                training_loss, training_accuracy = trainer.train(x,y, task_id=None)
                metric_monitor.update_averages(args.models, training_loss, training_accuracy)

            metric_monitor.update_metrics(args.models, 'train', task_id, num_batches=len(loader_train), reset_averages=True)

            if args.monitor:
                if (monitor_last_epoch and epoch == args.epochs) or (not monitor_last_epoch):
                    monitors.plots.plot_weights(writer, args.models, model, task_id, epoch)
                    monitors.plots.plot_gradients(writer, args.models, model, task_id, epoch)

            ######## ACO VALIDATION ########
            if is_aco:
                if task_id == 0:
                    aco_val_loss = metric_monitor.get_metric(args.models, 'train', 'loss', task_id)
                else:
                    aco_val_loss = 0.
                    num_val_samples = 0
                    loader_aco_val = rehe_val.memory_loader(batch_size=512, shuffle=False, drop_last=False)
                    for x,y in loader_aco_val:
                        x,y = x.to(device), y.to(device)
                        val_loss, _ = trainer.test(x,y)
                        val_batch_size = y.size()[0]
                        aco_val_loss += val_loss * val_batch_size
                        num_val_samples += val_batch_size
                    aco_val_loss /= num_val_samples
                aco.pheromone_update(ant_path, args.aco_reward_coef / aco_val_loss, update_rule=args.aco_update_rule, evaporation_rate=args.aco_evaporation_rate)
                if aco_val_loss <= best_aco_val_loss:
                    best_aco_val_loss = aco_val_loss
                    best_masks_so_far = tensor_masks
                pheromone_stat_logger.log_stat(aco.pheromone_stat())

            ######## VALIDATION ########
            for x,y in loader_task_test:
                x,y = x.to(device), y.to(device)

                validation_loss, validation_accuracy = trainer.test(x,y, task_id=None)
                metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

            metric_monitor.update_metrics(args.models, 'val', task_id, num_batches=len(loader_task_test), reset_averages=True)
            train_acc = metric_monitor.get_metric(args.models, 'train', 'acc', task_id)
            val_acc = metric_monitor.get_metric(args.models, 'val', 'acc', task_id)
            train_loss = metric_monitor.get_metric(args.models, 'train', 'loss', task_id)
            val_loss = metric_monitor.get_metric(args.models, 'val', 'loss', task_id)
            result_logger.warning(
                f"{args.models},"
                f"{task_id},{epoch}," 
                f"{train_acc},"
                f"{val_acc},"
                f"{train_loss},"
                f"{val_loss}"
            )
            step = task_id * args.epochs + epoch
            mlflowLogger.log_metric('train_acc', train_acc, step=step)
            mlflowLogger.log_metric('val_acc', val_acc, step=step)
            mlflowLogger.log_metric('train_loss', train_loss, step=step)
            mlflowLogger.log_metric('val_loss', val_loss, step=step)
        ######## END OF CURRENT TASK ########

        rehe.record_patterns(loader_task_train)
        if is_aco:
            rehe_val.record_patterns(loader_task_train)
            model.set_masks(best_masks_so_far)
            loader_mem = rehe.memory_loader(batch_size=args.batch_size, drop_last=False)
            for _ in range(args.end_task_epochs):
                for id_batch, (x,y) in enumerate(loader_mem):
                    if args.rehe_patterns > 0:
                        x, y = rehe.concat_to_batch(x,y)
                    x,y = x.to(device), y.to(device)
                    training_loss, training_accuracy = trainer.train(x,y, task_id=None)
            weight_ratio_logger_val.log(model.get_weight_ratios())

        if args.save:
            clutils.experiments.utils.save_model(model, args.models, result_folder, version=str(task_id))
            if is_aco:
                aco.save(os.path.join(result_folder, 'pheromone'), f'pheromone_{task_id}')

        if task_id < args.n_tasks - 1:

            if args.expand_output > 0:
                model.expand_output_layer(n_units=args.expand_output)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                trainer.optimizer = optimizer

        ######## INTERMEDIATE TEST ########
        # DO THIS TEST FOR ALL PREVIOUS TASKS, IF ARGS.NOT_INTERMEDIATE_TEST IS DISABLED
        # OR FOR THE LAST TASK ONLY, IF ARGS.NOT_TEST IS DISABLED

        if not args.not_intermediate_test or ( (not args.not_test) and (task_id == args.n_tasks-1) ):
            avg_test_loss = 0
            avg_test_acc = 0
            for intermediate_task_id in range(task_id+1):

                _, loader_task_test = dataset.get_task_loaders(task_id=intermediate_task_id)

                for x,y in loader_task_test:
                    x,y = x.to(device), y.to(device)

                    intermediate_loss, intermediate_accuracy = trainer.test(x,y, task_id=None)
                    metric_monitor.update_averages(args.models, intermediate_loss, intermediate_accuracy)

                metric_monitor.update_intermediate_metrics(args.models, len(loader_task_test), task_id, intermediate_task_id)
                result_logger.info(f"Intermediate accuracy {args.models}:"
                    f"{task_id} - {intermediate_task_id}:"
                    f"{metric_monitor.intermediate_metrics[args.models]['acc'][intermediate_task_id, task_id]}")
                
                avg_int_loss = metric_monitor.intermediate_metrics[args.models]['loss'][intermediate_task_id, task_id]
                avg_int_acc = metric_monitor.intermediate_metrics[args.models]['acc'][intermediate_task_id, task_id]
                avg_test_loss += avg_int_loss
                avg_test_acc += avg_int_acc
                mlflowLogger.log_metric(f'test_loss_task_{intermediate_task_id}', avg_int_loss, step=task_id)
                mlflowLogger.log_metric(f'test_acc_task_{intermediate_task_id}', avg_int_acc, step=task_id)
            avg_test_loss /= (task_id+1)
            avg_test_acc /= (task_id+1)
            mlflowLogger.log_metric(f'test_loss', avg_test_loss, step=task_id)
            mlflowLogger.log_metric(f'test_acc', avg_test_acc, step=task_id)

    if args.monitor:
        writer.close()

    result_logger.info("Saving results...")
    metric_monitor.save_intermediate_metrics()
    monitors.plots.plot_learning_curves(args.models, result_folder)
    result_logger.info("Done!")

    mlflowLogger.end_run()
