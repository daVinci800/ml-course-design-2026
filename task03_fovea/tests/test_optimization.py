import unittest
import numpy as np
from fovea.train import early_stopping_ready,epoch_sigma
from fovea.ensemble import candidate_recipes,combine_predictions


class OptimizationTests(unittest.TestCase):
    def test_minimum_epochs_allow_complete_cosine_and_sigma_schedule(self):
        settings=dict(patience=10,min_epochs=100,epochs=100,sigma=1.7,sigma_end=.8)
        self.assertFalse(early_stopping_ready(30,settings,36))
        self.assertFalse(early_stopping_ready(30,settings,99))
        self.assertTrue(early_stopping_ready(30,settings,100))
        self.assertAlmostEqual(epoch_sigma(settings,99),.8)
        self.assertTrue(early_stopping_ready(10,dict(patience=10),36))
        self.assertFalse(early_stopping_ready(9,dict(patience=10),100))

    def test_fixed_recipes_and_weighted_original_pixel_ensemble(self):
        recipes=candidate_recipes()
        self.assertEqual(len(recipes),12)
        for recipe in recipes.values(): self.assertAlmostEqual(sum(recipe.values()),1)
        a={1:dict(xy=[100,200]),2:dict(xy=[300,400])}
        b={1:dict(xy=[200,300]),2:dict(xy=[400,500])}
        combined=combine_predictions({'a':a,'b':b},{'a':1,'b':3})
        np.testing.assert_allclose(combined[1]['xy'],[175,275])
        np.testing.assert_allclose(combined[2]['xy'],[375,475])

    def test_ensemble_rejects_missing_ids_nonfinite_and_bad_weights(self):
        a={1:dict(xy=[100,200])}
        with self.assertRaises(ValueError): combine_predictions({'a':a},{'missing':1})
        with self.assertRaises(ValueError): combine_predictions({'a':a,'b':{}},{'a':.5,'b':.5})
        with self.assertRaises(ValueError): combine_predictions({'a':a},{'a':-1})
        with self.assertRaises(ValueError): combine_predictions({'a':{1:dict(xy=[float('nan'),0])}},{'a':1})


if __name__=='__main__': unittest.main(verbosity=2)
