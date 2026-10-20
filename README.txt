# Not quite a Readme yet

## Let's say "journal"

### What am I doing?


    - wrote the architecture for 2 base CNN AE, one for phase, one for amplitude. Separate training. (13-14/10)
    - With optuna looked for the best hyprparams for both (14/10 night)
    - training them on the kinda new dataset NRHybSur3dq8 on artemisa ( 15/10 morning-afternoon)
    - Prepared script to analyze the latent space representation of the two AEs. Waiting for training (15/10 afternoon).
    - Written one single autoEncoder for both amplitude and phase (should come with benefits) (15/10 afternoon).
    - Step back: the first two small AE (one for amplitude and one for Phase), didn't perform as i wished. In particular the phase autoencoder had very bad performance (like more than 1e-3 loss after 3000 epochs) and I think the reason is that i set weight decay very high. This hypothesis should be confirmed by the fact that weights on output layer to the latent space are veeery small (less than 1e-3) so I think that regularization here killed learning. Note for future self: optuna hyperparams suggestion need to be checked better.
    - Look for best hyprparams for Double architecture. Done, will restart training on 16 afternoon bc eve here weight regularization could be too high.
    - Train the DOUBLE architecture on the first dataset too. As above: train started on 16 afternoon.
    - Train for the 2small architectures start on 16 afternoon too.
    - Start secondo phase of training on both approaches??? idk. let's see.

    - 18-19/10: re-did the architecture of AE, better and a little bigger, now will try adam VS soap,
    - 20/10 : tried adam vs soap, soap after 16 hours is still at 1600 epoch (adam already finished), but from a checkpoint model i looked at the mismatch distribution and is more tight than adam counterpart. I hope that in the next ~ 1300 epoch will lower loss under 1e-4.
    - 20/10 still: implemented Mismatch common loss of tasos, should work w/o Nans, since one loss for 2 Networks seems a degeneracy to me, i keep phase and validation loss along the common mismatch one, waiting for better ideas. Launched 2 training to benchmark soap+MAEsmatch (i called it like this) and adam+MAEsmatch.
    - Maybe in the next days i can try to apply the other suggestion of tasos that is scrap the phase (i hate it) in favor of the frequency. 
