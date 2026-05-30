from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

def evaluate_predictions(y_true, y_pred, languages=None):

    # Računa ukupni Macro F1 i Accuracy, te Macro F1 zasebno za svaki jezik ako su jezici navedeni.
   
    metrics = {}
    
    # Ukupne performanse
    metrics['overall_macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Ako imamo podatke o jezicima, računamo Macro F1 za svaki jezik pojedinačno
    if languages is not None:
        lang_data = defaultdict(lambda: {'true': [], 'pred': []})
        
        # Grupiraj predikcije po jeziku
        for yt, yp, lang in zip(y_true, y_pred, languages):
            lang_data[lang]['true'].append(yt)
            lang_data[lang]['pred'].append(yp)
            
        metrics['per_language'] = {}
        for lang, data in lang_data.items():
            lang_macro_f1 = f1_score(data['true'], data['pred'], average='macro')
            lang_acc = accuracy_score(data['true'], data['pred'])
            metrics['per_language'][lang] = {
                'macro_f1': lang_macro_f1,
                'accuracy': lang_acc
            }
            
    return metrics

def print_evaluation_report(metrics):

    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Overall Macro F1: {metrics['overall_macro_f1']:.4f}")
    
    if 'per_language' in metrics:
        print("\nPerformance per language:")
        for lang, lang_metrics in metrics['per_language'].items():
            print(f"  - [{lang.upper()}] Macro F1: {lang_metrics['macro_f1']:.4f} | Accuracy: {lang_metrics['accuracy']:.4f}")
    print("===================================================\n")
