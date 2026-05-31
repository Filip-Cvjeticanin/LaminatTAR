def run_experiment(experiment_data):
    # Get the experiment setup
    log_path = experiment_data["log_path"]
    seed = experiment_data["seed"]
    train_languages = experiment_data["train_languages"]
    test_seen_languages = experiment_data["test_seen_languages"]
    test_unseen_languages = experiment_data["test_unseen_languages"]
    use_sythetic = experiment_data["use_sythetic"]
    use_sent_augmentation = experiment_data["use_sent_augmentation"]



def compare_augmentation_methods():
    experiment1 = {

    }
    
    experiment2 = {

    }

    experiment3 = {

    }

    experiments = [
        experiment1,
        experiment2,
        experiment3
    ]

    for e in experiments:
        run_experiment(e)