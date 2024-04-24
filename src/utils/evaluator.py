import pandas as pd
from typing import Any, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class Evaluator:
    """Evaluator."""
    
    def __init__(self, df: pd.DataFrame, across_class_names: bool = False, averaging: Union[str, None] = None) -> None:
        """class constructor

        Args:
            df (pd.DataFrame): dataframe containing the evaluation results.

            across_class_names (bool, optional): Whether to split the dataset and evaluate each 
            portion (e.g Dialectical or Standard). Defaults to False.
        """
        
        # dataframe object
        self.df = df
        
        # evaluate across class type
        self.across_class_names = across_class_names
        
        # class label order
        self.type_labels = [0, 1, 2]
        
        # class names
        self.classname_class_mapping = {'0': "standard", '1': "dialectical"}
        
        # averaging
        self.average = averaging
    
    def evaluate(self, df: pd.DataFrame) -> Tuple:
        """Evaluate the given results.

        Args:
            df (pd.DataFrame): dataframe containing the evaluation results or a portion of the dataset.
            
        Returns:
            Tuple: Evaluation results.
        """

        # change dtype of columns
        df.loc[:, 'gt_type'] = df['gt_type'].astype(int)
        df.loc[:, 'pred_type'] = df['pred_type'].astype(int)

        # compute the accuracy
        accuracy = accuracy_score(df['gt_type'], df['pred_type'])
        
        # Compute precision
        precision = precision_score(df['gt_type'], df['pred_type'], average = self.average, labels = self.type_labels)
    
        # Compute recall
        recall = recall_score(df['gt_type'], df['pred_type'], average = self.average, labels = self.type_labels)
        
        # Compute F1 score
        f1 = f1_score(df['gt_type'], df['pred_type'], average = self.average, labels = self.type_labels)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(df['gt_type'], df['pred_type'], labels = self.type_labels)

        # Compute classification report
        class_report = classification_report(df['gt_type'], df['pred_type'], labels = self.type_labels)
        
        return {
            'accuracy': accuracy, 
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_matrix': conf_matrix,
            'class_report': class_report
        }
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """_summary_

        Returns:
            Any: dict containing the evaluation results (for the overall dataset, or portions across class_name values)
        """
        
        # evalute a cross class names
        if self.across_class_names:

            # results
            results = {}
        
            # get unique class names
            class_names_ft = self.df['class_name'].unique()
            
            # evaluate across each class
            for class_name in class_names_ft:
                
                # convert numeric class label to class name
                class_name_label = self.classname_class_mapping.get(str(class_name))

                # filter DataFrame for the current class
                class_df = self.df[self.df['class_name'] == class_name]
                
                # evaluate
                class_df_evaluation = self.evaluate(class_df)
                
                # append to the results object
                results[class_name_label] = class_df_evaluation
        
            return results
        
        return self.evaluate(self.df)
