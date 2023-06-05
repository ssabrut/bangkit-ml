import pandas as pd
import numpy as np
import tensorflow as tf

class PricePredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = tf.keras.models.load_model('common/models/price_model.h5')

    def tf__fill_ones(self):
        with ag__.FunctionScope(
            "fill_ones",
            "fscope",
            ag__.ConversionOptions(
                recursive=True,
                user_requested=True,
                optional_features=(),
                internal_convert_user_code=True,
            ),
        ) as fscope:
            ag__.ld(self).data.loc[
                ag__.ld(self).data["Luas panen (ha)"] < 0, "Luas panen (ha)"
            ] = 1
            ag__.ld(self).data.loc[
                ag__.ld(self).data["produksi (kuintal)"] < 0, "produksi (kuintal)"
            ] = 1

    def tf__one_hot(self):
        with ag__.FunctionScope(
            "one_hot",
            "fscope",
            ag__.ConversionOptions(
                recursive=True,
                user_requested=True,
                optional_features=(),
                internal_convert_user_code=True,
            ),
        ) as fscope:
            ag__.ld(self).data = ag__.converted_call(
                ag__.ld(pd).get_dummies,
                (ag__.ld(self).data,),
                dict(columns=["Daerah", "tanaman"]),
                fscope,
            )

    def tf__split_and_transform(self):
        with ag__.FunctionScope(
            "split_and_transform",
            "fscope",
            ag__.ConversionOptions(
                recursive=True,
                user_requested=True,
                optional_features=(),
                internal_convert_user_code=True,
            ),
        ) as fscope:
            do_return = False
            retval_ = ag__.UndefinedReturnValue()
            X_num = (
                ag__.ld(self)
                .data.loc[:, ["Luas panen (ha)", "produksi (kuintal)"]]
                .values
            )
            X_cat = ag__.ld(self).data.iloc[:, 4:].values
            X_transform = ag__.converted_call(
                ag__.ld(np).log, (ag__.ld(X_num),), None, fscope
            )
            X = ag__.converted_call(
                ag__.ld(np).concatenate,
                ([ag__.ld(X_cat), ag__.ld(X_transform)],),
                dict(axis=1),
                fscope,
            )
            try:
                do_return = True
                retval_ = ag__.ld(X)
            except:
                do_return = False
                raise
            return fscope.ret(retval_, do_return)

    def tf__pipeline(self):
        with ag__.FunctionScope('pipeline', 'fscope',
                                ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(),
                                                       internal_convert_user_code=True)) as fscope:
            do_return = False
            retval_ = ag__.UndefinedReturnValue()
            ag__.converted_call(ag__.ld(self).tf__fill_ones, (), None, fscope)
            ag__.converted_call(ag__.ld(self).tf__one_hot, (), None, fscope)
            X = ag__.converted_call(ag__.ld(self).tf__split_and_transform, (), None, fscope)
            try:
                do_return = True
                retval_ = ag__.ld(X)
            except:
                do_return = False
                raise
            return fscope.ret(retval_, do_return)

    def tf__get_prediction(self):
        with ag__.FunctionScope('get_prediction', 'fscope',
                                ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(),
                                                       internal_convert_user_code=True)) as fscope:
            do_return = False
            retval_ = ag__.UndefinedReturnValue()
            X = ag__.converted_call(ag__.ld(self).tf__pipeline, (), None, fscope)
            yhat = ag__.converted_call(ag__.ld(self).model.predict, (ag__.ld(X),), None, fscope)
            try:
                do_return = True
                retval_ = ag__.converted_call(ag__.ld(np).exp, (ag__.ld(yhat),), None, fscope)
            except:
                do_return = False
                raise
            return fscope.ret(retval_, do_return)