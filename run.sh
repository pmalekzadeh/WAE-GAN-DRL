# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/var_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/var_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/var_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/var_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True

# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/cvar_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/cvar_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/cvar_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=long_dip/rl/cvar_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True


# python3 run.py -delta_hedging=True -logger_prefix=short_dip/rl/meanstd_95/1.02 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.2 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/rl/meanstd_95/1.04 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.4 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/rl/meanstd_95/1.06 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.6 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/rl/meanstd_95/1.08 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True

# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95/1.02 -eval_sim=5000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95/1.04 -eval_sim=5000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95/1.06 -eval_sim=5000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95/1.08 -eval_sim=5000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0

# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/cvar_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/cvar_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/cvar_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/cvar_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0

# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/var_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0

# python3 greek_run.py -logger_prefix=long_dip/delta/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -gbm=True -poisson_rate=0.0 
# python3 greek_run.py -logger_prefix=long_dip/delta/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -gbm=True -poisson_rate=0.0 
# python3 greek_run.py -logger_prefix=long_dip/delta/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -gbm=True -poisson_rate=0.0 
# python3 greek_run.py -logger_prefix=long_dip/delta/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 

# python3 one_day_optimal_run.py -logger_prefix=short_dip/no_hedge/cvar_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=-1 -barrier_strike=10.2 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/no_hedge/cvar_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=-1 -barrier_strike=10.4 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/no_hedge/cvar_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=-1 -barrier_strike=10.6 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/no_hedge/cvar_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True

# python3 one_day_optimal_run.py -logger_prefix=long_dip/no_hedge/cvar_95_10000/1.02 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.2 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=long_dip/no_hedge/cvar_95_10000/1.04 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.4 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=long_dip/no_hedge/cvar_95_10000/1.06 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.6 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -no_hedge=True
# python3 one_day_optimal_run.py -logger_prefix=long_dip/myopic/cvar_95_10000/1.08 -eval_sim=10000 -S0=10.6 -long_short=1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0

# ROBUSTNESS TESTS

# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/cvar_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_cvar -init_vol_eval=0.1
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/cvar_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_cvar -init_vol_eval=0.2
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/cvar_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_cvar -init_vol_eval=0.3
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/cvar_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_cvar -init_vol_eval=0.4
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/cvar_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_cvar -init_vol_eval=0.5

# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/var_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_var -init_vol_eval=0.1
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/var_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_var -init_vol_eval=0.2
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/var_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_var -init_vol_eval=0.3
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/var_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_var -init_vol_eval=0.4
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/var_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_var -init_vol_eval=0.5

# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/meanstd_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_meanstd -init_vol_eval=0.1
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/meanstd_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_meanstd -init_vol_eval=0.2
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/meanstd_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_meanstd -init_vol_eval=0.3
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/meanstd_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_meanstd -init_vol_eval=0.4
# python3 run.py -delta_hedging=True -logger_prefix=short_dip/robust/rl/meanstd_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -scale_action=True -eval_only=True -agent_path=logs/policies/1.08_dip_meanstd -init_vol_eval=0.5

# python3 greek_run.py -logger_prefix=short_dip/robust/delta/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 -init_vol=0.1
# python3 greek_run.py -logger_prefix=short_dip/robust/delta/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 -init_vol=0.2
# python3 greek_run.py -logger_prefix=short_dip/robust/delta/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 -init_vol=0.3
# python3 greek_run.py -logger_prefix=short_dip/robust/delta/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 -init_vol=0.4
# python3 greek_run.py -logger_prefix=short_dip/robust/delta/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -gbm=True -poisson_rate=0.0 -init_vol=0.5

# python3 one_day_optimal_run.py -logger_prefix=short_dip/myopic/meanstd_95/1.02 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.2 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/myopic/meanstd_95/1.04 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.4 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/myopic/meanstd_95/1.06 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.6 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/myopic/meanstd_95/1.08 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -larger_search_range=True

# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/cvar_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.1 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/cvar_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.2 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/cvar_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.3 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/cvar_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.4 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/cvar_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=cvar -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.5 -larger_search_range=True

# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/var_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.1 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/var_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.2 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/var_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.3 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/var_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.4 -larger_search_range=True
# python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/var_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=var -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.5 -larger_search_range=True

python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/meanstd_95/1.08_10 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.1 -larger_search_range=True
python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/meanstd_95/1.08_20 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.2 -larger_search_range=True
#python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/meanstd_95/1.08_30 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.3 -larger_search_range=True
python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/meanstd_95/1.08_40 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.4 -larger_search_range=True
python3 one_day_optimal_run.py -logger_prefix=short_dip/robust/myopic/meanstd_95/1.08_50 -eval_sim=5000 -S0=10.6 -long_short=-1 -barrier_strike=10.8 -obj_func=meanstd -threshold=0.95 -gbm=True -poisson_rate=0.0 -init_vol=0.5 -larger_search_range=True