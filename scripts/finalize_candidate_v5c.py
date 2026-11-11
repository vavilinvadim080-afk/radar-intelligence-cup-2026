"""Select on tune-only CV, refit both heads, then permit a separate explicit freeze."""
import shutil
from scripts import train_service_v5c as t
from scripts import probe_matching_v5c as matching
from scripts import probe_reply_v5c as reply


def main():
    t.assert_unfrozen();t.verify_baseline()
    sources,code=t.training_sources(),t.semantic_code_hashes()
    old=t.OUT/'tune-grouped-results.json'
    historical=t.OUT/'tune-grouped-before-matching.json'
    if old.exists() and not historical.exists():
        shutil.copyfile(old,historical)
    needs_cv=matching.main()
    reply_cv=reply.main()
    store=t.load_features('tune');gold,reply_gold=t.load_tune_gold(store)
    rows=t.unique_needs(store.data['needs_inputs']);pairs=t.training_pairs(store,rows,gold)
    chosen=needs_cv['selected'];reply_chosen=reply_cv['selected']
    heads=t.fit_heads(store,rows,gold,pairs,list(range(len(rows))),list(range(len(pairs))),
                      chosen['kind_C'],chosen['pair_C'])
    heads['reply_pair']=reply.fit_reply_head(pairs,reply.reply_examples(store,store.data['reply_inputs'],reply_gold),
                                             reply_chosen['transform'],reply_chosen['C'],reply_chosen['reply_weight'])
    provenance=dict(training_sources_sha256=sources,semantic_code_sha256=code,
                    feature_source_sha256=t.sha256_file(t.OUT/'tune-features.json'),
                    preparation_contract_sha256=t.feature_contract(),
                    models=t.plain_metadata(store.data['metadata']),
                    training_ids=[r['id'] for r in rows],
                    training_families=sorted({t.family_key(r,gold) for r in rows}),
                    semantic_reply_supervision='Tune same_topic only; no permission/lifecycle targets')
    heads['provenance']=provenance
    if sources!=t.training_sources() or code!=t.semantic_code_hashes():
        raise RuntimeError('Tune sources or code changed during final refit')
    t.save_checkpoint(t.HEADS,heads)
    config=dict(version='5c',selected_backend='qwen',status='tuned_control_pending',
                needs=chosen['config'],reply=reply_chosen['config'],
                training=dict(feature_source_sha256=provenance['feature_source_sha256'],
                              heads_sha256=t.sha256_file(t.HEADS)),
                selection='Tune whole-family CV selects history grouping; no control labels.',
                reply_selection='Whole reference-component CV: zero ordinary false positives first, then recall.')
    features=t.CachedCandidateFeatures(store,heads)
    predicted=t.CandidateSelector(features,backend='qwen',config=config['needs']).predict(store.data['needs_inputs'])
    replies=t.predict_replies(t.CandidateReplyPolicy(features,config=config['reply']),store.data['reply_inputs'])
    baseline=t.baseline_predictions(store)
    report=dict(status='tuned',needs_cv=needs_cv,reply_cv=reply_cv,training=provenance,
                needs_resubstitution=t.score_needs(predicted,gold),
                reply_resubstitution=dict(config=config['reply'],metrics=t.score_replies(replies,store.data['reply_inputs'],reply_gold),
                    evaluation='Refit on all tune needs/reply semantic labels; this is resubstitution, not acceptance.'),
                baseline_tune=dict(needs=t.score_needs(baseline['needs'],gold),
                                   reply=t.score_replies(baseline['reply'],store.data['reply_inputs'],reply_gold)),
                auto_enable=False,caveat='Tune CV selected the candidate. Independent control remains sealed. '
                'History regrouping may revise earlier observed groups; no online latency is claimed.')
    t.save_checkpoint(t.CONFIG,config)
    t.save_checkpoint(t.OUT/'tune-grouped-results.json',report)
    t.check_training_provenance(heads,config,store)
    print(t.json.dumps(dict(needs_cv=chosen['metrics'],reply_cv=reply_chosen['metrics'],
                           reply_refit=report['reply_resubstitution']['metrics']),indent=2))


if __name__=='__main__':
    main()
