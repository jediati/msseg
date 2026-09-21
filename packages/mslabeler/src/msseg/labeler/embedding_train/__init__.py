"""Offline training of the task-free region encoder (``docs/design_region_autoencoder.md``).

* ``shards``  -- the harvest on disk: ``harvest.json`` + one ``.npz`` per
                 (item, persistence) holding the statistics rows and the region
                 arcs; a writer the harvest drivers share and a reader the
                 trainer uses. numpy only.
* ``model``   -- the torch nets (encoder, projection head, reconstruction
                 head) and the PCA baseline (numpy).
* ``train``   -- the objective and the loop: random-walk InfoNCE over the
                 region graph + reconstruction + variance/covariance, with
                 group dropout and marginal corruption as the augmentation;
                 produces an ``embedding.EncoderBundle``.
* ``cli``     -- the ``train`` sub-command's arguments, shared by
                 ``mspath-embed`` (and a coupon driver later).

torch is imported only inside ``train`` when the architecture needs it; the
PCA path and the shard format work on the pure-Python wheel.
"""
