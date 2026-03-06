class Config:
    embed_dim = 300
    window_size = 10
    neg_samples = 5
    min_count = 5
    subsample_t = 1e-5
    lr_start = 0.025
    lr_min = 0.0001
    epochs = 20
    batch_size = 4096
    chunk_size = 500000      # tokens per processing chunk
    neg_table_size = 10000000
    seed = 1
    data_dir = 'data'
    results_dir = 'results'

    # Adaptive Frequency-Based Window Sizing
    use_afws = False
    afws_min_window = 3
    afws_max_window = 15
    afws_alpha = 0.5

    def to_dict(self):
        return {k: v for k, v in vars(type(self)).items()
                if not k.startswith('_') and not callable(v)}
