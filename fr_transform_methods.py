import numpy as np 


class fr_transforms:
    def __init__(self, tot_fr_df, transform_method="identity", **kwargs):
        self.tot_fr_df = tot_fr_df
        self.cstr_scale = kwargs.get('cstr_scale', 0)
        self.fit_type = kwargs.get('fit_type', 'all_to_all')
        self.transform_method = transform_method

    def make_cstr_input_matrix(self, **kwargs):
        if self.transform_method == "identity":
            unit_id = kwargs.get('unit_id', None)
            if unit_id is None:
                raise ValueError
            y_shape = kwargs.get('y_shape', None)
            if y_shape is None:
                raise ValueError
            return self.identity_cstr(self.tot_fr_df, unit_id, y_shape)

    def identity_cstr(self, tot_fr_df, unit_id, y_shape):
        print(tot_fr_df.shape)
        X_cstr = np.array([]).reshape(tot_fr_df.shape[1], 0)
        if self.cstr_scale != 0:
            df_X = self.tot_fr_df.T.copy()
            if self.fit_type == 'all_to_all':
                df_X[unit_id] = np.zeros(y_shape)
                X_cstr = self.cstr_scale*np.array(df_X)

            del df_X
        
        print('Cstr shape is: {}'.format(X_cstr.shape))
        return X_cstr