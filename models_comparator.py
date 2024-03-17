from train_loop import train_loop
import matplotlib.pyplot as plt

class ModelsComparator:
    def __init__(self, models_settings_to_compare, train_dataloader):
        self.models_settings = models_settings_to_compare
        self.dataloader = train_dataloader
    
    def compare_models(self, epochs=25, device='cpu'):
        results = []
        for model, optimizer, loss_fn in self.models_settings:
            model.to(device)
            loss_fn.to(device)

            result = train_loop(
                self.dataloader,
                model,
                optimizer,
                loss_fn,
                epochs=epochs,
                device=device
            )

            results.append({'model': model.__class__.__name__, 'results': result})

        self.comparison_results = results
        return results

    def plot_comparison_results(self):
        if self.comparison_results and len(self.comparison_results) > 0:

            for model_results in self.comparison_results:
                model_name = model_results['model']
                results = model_results['results']
                epochs = [entry['epoch_num'] for entry in results]
                losses = [entry['avg_loss'] for entry in results]

                # Plotting
                plt.plot(epochs, losses, label=model_name)
                plt.title('Average Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Average Loss')
                plt.grid(True)

            plt.legend()
            plt.show()