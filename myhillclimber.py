class HillClimber:
    "This class develops the Hill Climber algorithm for the provided datasets"

    def __init__(self):
        self.ScoreMetric = utils.ScoreMetric

    def DoHillClimb(
        self, 
        target:str,
        direction:str,
        cutoff:float,
        neg_wgt:str,
        OOF_Preds: pd.DataFrame,
        Mdl_Preds: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ):
        """
        This method performs hill-climbing on the OOF and Test predictions dataset and returns the below-
        1. OOF ensemble predictions
        2. Test set predictions
        3. Score dataframe (with scores in sort-order)
        """

        oof_df     = OOF_Preds
        test_preds = Mdl_Preds
    
        # Scoring the individual models:-
        Scores = pd.DataFrame(index = oof_df.columns, columns = ['Score'])
    
        for col in oof_df.columns:
            Scores.at[col, 'Score'] = self.ScoreMetric(y, oof_df[col].values.flatten())
    
        # Sorting scores
        Scores.sort_values(
            by= 'Score',
            ascending = [True if direction == 'minimize' else False],
            inplace = True,
        )
    
        PrintColor(f"\n----- Data preparation: ------ \n");
        display(
            Scores.
            transpose().
            style.
            format(precision = 5)
            )
    
        PrintColor(f"\n ----- Initiating hill-climb ----- \n");
        STOP = False
        current_best_ensemble   = oof_df.iloc[:,0]
        current_best_test_preds = test_preds.iloc[:,0]
        MODELS                  = oof_df.iloc[:,1:]
    
        if neg_wgt == "Y":
            weight_range = np.arange(-0.5,0.51,0.01);
        else:
            weight_range = np.arange(0.01,0.51,0.01);
    
        history = [self.ScoreMetric(y, current_best_ensemble)]
    
        i=0
    
        # Hill climbing algorithm:-
        while not STOP:
            i+=1
    
            potential_new_best_cv_score = self.ScoreMetric(y, current_best_ensemble)
            k_best, wgt_best = None, None
    
            for k in MODELS:
                for wgt in weight_range:
                    potential_ensemble = (1- wgt) * current_best_ensemble + wgt * MODELS[k]
                    cv_score = self.ScoreMetric(y, potential_ensemble)
    
                    if direction == 'minimize':
                        if cv_score < potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
                    if direction == 'maximize':
                        if cv_score > potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
            if k_best is not None:
                current_best_ensemble   = (1- wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
                current_best_test_preds = (1- wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
                MODELS.drop(k_best, axis=1, inplace=True)
    
                if MODELS.shape[1]==0:  STOP = True
    
                num_space = 50 - len(k_best) if i <= 9 else 49 - len(k_best)
                PrintColor(f" {i}.{k_best} {' ' * num_space} Weight = {wgt_best: .4f} {' ' * 5} Score = {potential_new_best_cv_score:.6f}",
                           color = Fore.CYAN
                          )
                del num_space
    
                history.append(potential_new_best_cv_score)
    
            else:
                STOP = True
    
        return (current_best_ensemble, current_best_test_preds, Scores)
