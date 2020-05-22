if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
    valid_loss = self.eval(self.valid_data)
    valid_ppl = math.exp(min(valid_loss, 100))
    print('Validation perplexity: %g' % valid_ppl)

    ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

    self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)