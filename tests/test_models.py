import unittest

from gadma import *

from .test_data import YRI_CEU_DATA

try:
    import dadi
    DADI_NOT_AVAILABLE = False
except ImportError:
    DADI_NOT_AVAILABLE = True

import numpy as np
import os

EXAMPLE_FOLDER = os.path.join('examples', 'YRI_CEU')

class TestModels(unittest.TestCase):
    def dadi_wrapper(self, func):
        def wrapper(param, ns, pts):
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = func(param, xx, phi)
            sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
            return sfs
        return wrapper

    def get_variables_for_gut_2009(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')
        return (nu1F, nu2B, nu2F, m, Tp, T, Dyn)

    def test_custom_dm_init(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                location = os.path.join(EXAMPLE_FOLDER, engine.id,
                                        'demographic_models.py')
                spec = importlib.util.spec_from_file_location("module",
                                                              location)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules['module'] = module
                func = getattr(module, 'model_func')
                variables = self.get_variables_for_gut_2009()[:-1] 
                dm = CustomDemographicModel(func, variables)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_0(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex([], ns, pts)

        dm = EpochDemographicModel()
        pts = [40, 50, 60]
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate([], ns, pts)
        self.assertTrue(np.allclose(got, real))
        self.assertEqual(dm.number_of_populations(), 1)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_1(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            T, nu = param
            phi = dadi.Integration.one_pop(phi, xx, T=T, nu=nu)
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        param = [1., 0.5]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex(param, ns, pts)

        T = TimeVariable('T1')
        nu = PopulationSizeVariable('nu2')
        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu])
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate(param, ns, pts)
        self.assertTrue(np.allclose(got, real))
        self.assertEqual(dm.number_of_populations(), 1)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_gut_2pop(self):
        """
        Check loglikelihood of the demographic model from the YRI_CEU
        example of dadi.
        """
        nu1F, nu2B, nu2F, m, Tp, T, Dyn = self.get_variables_for_gut_2009()

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', 'Exp'])

        dic = {'nu1F': 1.880, 'nu2B': 0.0724, 'nu2F': 1.764, 'm': 0.930,
               'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp'}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = [dic[var.name] for var in dm.variables]
        ll = d.evaluate(values, pts=[40, 50, 60])
        self.assertEqual(int(ll), -1066)
        self.assertEqual(dm.number_of_populations(), 2)

    def test_fix_vars(self):
        nu1F, nu2B, nu2F, m, Tp, T, Dyn = self.get_variables_for_gut_2009()
        Dyn2 = DynamicVariable('SudDyn')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F], dyn_args=[Dyn2])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])

        dic = {'nu1F': 1.880, nu2B: 0.0724, 'nu2F': 1.764, 'm': 0.930,
               'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp', 'SudDyn': 'Sud'}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = dic#[dic[var.name] for var in dm.variables]
        ll1 = d.evaluate(values, pts=[40, 50, 60])
        n_par_before = dm.get_number_of_parameters(dic)
        
        dm.fix_variable(Dyn, 'Exp')
        d.model = dm
        ll2 = d.evaluate(dic, pts=[40, 50, 60])
        n_par_after = dm.get_number_of_parameters(dic)

        dm.unfix_variable(Dyn)
        n_par_after_after = dm.get_number_of_parameters(dic)

        self.assertEqual(ll1, ll2)
        self.assertEqual(n_par_before, 8)
        self.assertEqual(n_par_before, n_par_after + 1)
        self.assertEqual(n_par_before, n_par_after_after)

        dm.fix_dynamics(dic)
        n_par_without_dyns = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_without_dyns + 2)

        dm.unfix_dynamics()
        n_par_with_dyns = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_with_dyns)

        dic['Dyn'] = 'Sud'
        n_par_sud_model = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_sud_model + 1)

        # check fail
        var = PopulationSizeVariable('nu3')
        self.assertRaises(ValueError, dm.fix_variable, var, 3)
        self.assertRaises(ValueError, dm.unfix_variable, var)

#        dm.events[2].set_value(nu2F, 1.0)
#        n_par_after = dm.get_number_of_parameters(dic)
#        self.assertEqual(n_par_sud_model, n_par_after + 1)
