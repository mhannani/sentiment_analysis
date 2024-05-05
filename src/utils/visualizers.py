from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.evaluator import Evaluator


class Visualizer:
    """class incorporating all helper function for visualisation"""
    
    def __init__(self, evaluator: Evaluator) -> None:
        """class constructor for Visualizer class.

        Args:
            evaluator (Evaluator): Evaluator class
        """
        
        # evaluator
        self.evaluator = evaluator
        
        # class label
        self.type_labels = self.evaluator.type_labels
        
        # averaging
        self.averaging = self.evaluator.average

        # evaluate across class names
        self.across_class_names = self.evaluator.across_class_names

        # self.classname_class_mapping
        self.classname_class_mapping = self.evaluator.classname_class_mapping

        # evaluate
        self.evaluation = self.evaluator()
        
        # Define colors for the bars
        self.colors_type = ['lightcoral', 'lightgrey', 'lightgreen']
        
        # get only two colors
        self.colors_type_class_name = self.colors_type[::len(self.colors_type)-1]

        # types labels
        self.type_labels = ['Negative', 'Neutral', 'Positive']
        
        # class name labels
        self.class_name_labels = ['Dialectical', 'Standard']
    
    @staticmethod
    def annotate_bars(ax: plt.Axes, bars: any, values: Union[np.float64, np.ndarray]) -> None:
        """Annotate bars in the given axes with values

        Args:
            ax (plt.axes._axes.Axes): ax
            bars (plt.container.BarContainer): bars list
            values (Union[np.float64, np.ndarray]): value for each bar as np.float64 or np.ndarray
        """
        
        # Check if the value is scalar
        if np.isscalar(values):  
            # create np.ndarray from the singular value
            values = np.array([values])
        # else:
        #     values = np.array(values)
        
        # annotate bars
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.2f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                            xytext=(0, 0),
                            textcoords="offset points",
                            weight='bold',
                            ha='center', va='bottom')

    def plot_single_metric_across_classnames(self, metric: str) -> None:
        """Plots the givem metric only across the classname and not the types.

        Args:
            metric (str): metric names. 
        """
        
        # Create a figure for the chart
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Iterate over the classes (standard and dialectal)
        for i, class_label_key in enumerate(self.classname_class_mapping.keys()):

            # Class label
            class_label = self.classname_class_mapping[class_label_key]

            # Get metric values for the current class
            metric_values = self.evaluation[class_label][metric]

            # Plot the bar chart for metric
            bars = ax.bar([i], metric_values, color=self.colors_type_class_name[i], label=class_label)
            
            # Annotate the bars with metric values
            self.annotate_bars(ax, bars, metric_values)

            
        # Set chart labels and legend
        ax.set_xlabel('Class name')
        ax.set_ylabel(f'{metric.capitalize()}')
        ax.set_title(f'{metric.capitalize()} for Arabic tweets')
        ax.set_xticks(np.arange(len(self.class_name_labels)))
        ax.set_xticklabels(self.class_name_labels)

        # Show the plot
        plt.show()

    def visualize(self, metric: str) -> Union[None, float]:
        """Visualize the given metric.

        Args:
            metric (str): metric name: eg. precision, recall, or f1.
        """
        
        # for overall dataset and averaging used
        if self.averaging is not None:
            if not self.across_class_names:

                # get accuracy
                metric_value = self.evaluation[metric]
                
                # print it out
                print(f"- {metric.capitalize()}: {metric_value}")
        
                return metric_value
        
        # if metric is accuracy then returning the value if it's across the overall dataset
        # plot the bars char if it's acorss classname(dialectical or standard)
        if metric == 'accuracy':
            if self.across_class_names:
                return self.plot_single_metric_across_classnames(metric)

            # get accuracy
            accuracy = self.evaluation['accuracy']

            # print it out
            print(f"Accuracy: {accuracy}")

            return accuracy

        # Check if the input metric is valid
        if metric not in ['precision', 'recall', 'f1', 'accuracy']:
            raise ValueError(f"Invalid metric: {metric}. Expected one of ['precision', 'recall' or 'f1'].")
        
        if self.averaging is None and self.across_class_names:
            # figure with two charts
            # metric for each class [dialectical and standard]
            # and for each type[negative(-1.0), neutral(0.0), and positive(1.0)]

            # Create a figure for the charts
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            
            # figure title
            fig.suptitle(f'{metric.capitalize()} for each class')
            
            # Iterate over the classes (standard and dialectal)
            for ax, class_label_key in zip(axs, self.classname_class_mapping.keys()):
                
                # class label
                class_label = self.classname_class_mapping[class_label_key]

                # Get metric values for the current class
                metric_values = self.evaluation[class_label][metric]

                # Plot the bar chart for metric
                bars = ax.bar(self.type_labels, metric_values, color = self.colors_type)
                ax.set_title(f'{metric.capitalize()} for {class_label.capitalize()} Arabic tweets')
                ax.set_xlabel('Types')
                ax.set_ylabel(f'{metric.capitalize()}')
                
                # Add metric values on top of each bar
                self.annotate_bars(ax, bars, metric_values)
                

            # Show the plot
            plt.show()
            
        elif self.averaging is not None and self.across_class_names:
            # figure with two charts
            # metric for each class  [dialectical and standard]
            # and one metric value [when weighted, macro, or micro used as average parameter]

            return self.plot_single_metric_across_classnames(metric)

        elif self.averaging is None and not self.across_class_names:
            # figure with one chart
            # metric for overall dataset
            # and for each type[negative(-1.0), neutral(0.0), and positive(1.0)]
            
            # Create a figure for the chart
            fig, ax = plt.subplots(figsize=(5, 5))

            # Get metric values for the overall dataset
            metric_values = self.evaluation[metric]

            # Plot the bar chart for metric
            bars = ax.bar(np.arange(len(self.type_labels)), metric_values, color=self.colors_type)

            # Annotate the bars with metric values
            self.annotate_bars(ax, bars, metric_values)

            # Set chart labels and legend
            ax.set_xlabel('Type name')
            ax.set_ylabel(f'{metric.capitalize()}')
            ax.set_title(f'{metric.capitalize()} for Arabic tweets')
            ax.set_xticks(np.arange(len(self.type_labels)))
            ax.set_xticklabels(self.type_labels)

            # Show the plot
            plt.show()
        
        else:  # self.averaging is not None and not self.across_class_names:
            # No visualization possible as we have only one value for the metric.
            print(f"Info: No visualization possible for the given metric. e.g: {metric}.")
            
            # print out the metric value requested
            print(f"{metric.capitalize()}: ", self.evaluation[metric])
            
            return self.evaluation[metric]
    
    def show_cfm(self) -> None:
            """Show confusion matrix(metrices)"""
            
            if self.across_class_names:
                # Create a figure for the confusion matrices
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Iterate over the classes (standard and dialectical)
                for ax, class_label_key in zip(axs, self.classname_class_mapping.keys()):
                    # class label
                    class_label = self.classname_class_mapping[class_label_key]

                    # Get confusion matrix for the current class
                    conf_matrix = self.evaluation[class_label]['conf_matrix']

                    # Plot confusion matrix as a heatmap
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'Confusion Matrix for {class_label} Arabic tweets')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')

                # Adjust layout and display the plot
                plt.tight_layout()
                plt.show()

            else:
                # Create a figure for the confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))

                # Get confusion matrix for the overall dataset
                conf_matrix = self.evaluation['conf_matrix']

                # Plot confusion matrix as a heatmap
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix for Arabic tweets')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')

                # Display the plot
                plt.show()


class MYCVisualizer(Visualizer):
    """MYC visualizer for Binary classification

    Args:
        Visualizer (Visualizer): Sentiment Analysis results visualizer
    """
    
    def __init__(self, evaluator: Evaluator) -> None:
        """class constructor for the MYCVisualizer

        Args:
            evaluator (Evaluator): Evaluator for the MYC dataset
        """
        
        # class the parent's __init__ class
        super().__init__(evaluator)
        
        # Define colors for the bars
        self.colors_type = ['lightcoral', 'lightgreen']
        
        # get only two colors
        self.colors_type_class_name = self.colors_type[::len(self.colors_type)-1]

        # types labels
        self.type_labels = ['Negative', 'Positive']
        
        
def visualize_frequencies(df: pd.DataFrame, column_name: str) -> None:
    """Visualize the given feature frequencies as pie chart

    Args:
        df (pd.DataFrame): dataframe
        column_name (str): The column name to visualize its frequency
    """
    
    # calculate the frequency of each unique value in the 'type' column
    type_counts = df[column_name].value_counts()

    # create a pie chart using Matplotlib
    plt.figure(figsize=(4, 4))
    
    # create pie chart
    plt.pie(type_counts, labels=[f'{index}: {count} ({count/len(df)*100:.1f}%)' for index, count in type_counts.items()], autopct=None, startangle=140)
    
    # set title
    plt.title(f'Distribution of Tweets {column_name}')
    
    # equali axis dimentsion - circle pie chart
    plt.axis('equal')

    # show the pie chart
    plt.show()
