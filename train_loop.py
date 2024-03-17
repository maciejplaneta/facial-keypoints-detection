from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, loss_fn, lr_scheduler=None, epochs=25, device='cpu', debug=False):
    model.to(device)
    loss_fn.to(device)
    model.train()

    avg_losses = []
    print(f"Starting training loop for model '{model.__class__.__name__}'. Epochs: {epochs}, batches: {len(dataloader)}")

    # loop over the dataset multiple times
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print('____________________')
            # print(f'Batch {i}, data size: {len(inputs)}')
            # forward pass
            outputs = model(inputs)

            # # calculate loss
            loss = loss_fn(outputs, labels)
            # print(f'Batch[{i}] loss: {loss}')
            running_loss += loss

            # zero the parameter gradients
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()
        
        # LR scheduler step
        if lr_scheduler:
            if debug:
                print(f"Last lr: {lr_scheduler.get_last_lr()}")
            lr_scheduler.step()
            
        avg_batch_loss = running_loss/len(dataloader)
        avg_losses.append({'epoch_num': epoch, 'avg_loss': avg_batch_loss.item()})
        if debug:
            print(f'Total epoch[{epoch + 1}] loss: {running_loss}, Average batch loss: {avg_batch_loss}')

    print(f'Finished training, last avg batch loss: {avg_batch_loss}')
    return avg_losses