"""QAOA portfolio optimisation package.

Modules
    config         -- all knobs (seeds, universe, model parameters)
    data_pipeline  -- prices -> mu, Sigma (yahoo / cache / synthetic fallback)
    qubo           -- portfolio objective as a QUBO (own derivation)
    ising          -- QUBO -> Ising Hamiltonian (own mapping) + SparsePauliOp
    brute_force    -- exact 2^N enumeration (ground truth)
    qaoa_circuit   -- hand-built p-layer QAOA ansatz
    optimize       -- COBYLA / SPSA angle search + evaluation metrics
    baselines      -- greedy, local search, annealing, Markowitz+rounding
    noise          -- noisy AerSimulator study (largest useful p)
    hardware       -- OPTIONAL real-IBM run (token from env var only)
    main           -- end-to-end pipeline: runs everything, writes figures
"""
__version__ = "1.0.0"
