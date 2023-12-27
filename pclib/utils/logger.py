from torch.utils.tensorboard import SummaryWriter

def get_writer(model, log_dir, optimiser):

    writer = SummaryWriter(log_dir=log_dir)

    # Log model graph
    writer.add_graph(model, torch.rand(1, 3, 224, 224))

    # Log optimiser
    writer.add_text('optimiser', str(optimiser))

    return writer