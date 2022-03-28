


def evaluate_model_bias(model, savepath):

    
    # load the tf news dataset
    data_loader = DataLoader()#news_sent = data_loader.load_news_dataset()
    
    
    # professions and company names as target words
    prof = data_loader.read_target_words_from_json('exp_data/target_words/professions.json', index=0)
    comp = data_loader.read_target_words_from_json('exp_data/target_words/companies.json')
    sent_test = data_loader.read_target_words_from_json('exp_data/sentiment/test_words.json')
    
    # bias attributes for gender, sentiment and religion
    gender_attr = data_loader.read_bias_attributes_from_json('exp_data/gender/definitional_pairs.json')
    religion_attr = data_loader.read_bias_attributes_from_json('exp_data/religion/definitional_pairs.json')
    sentiment_attr = data_loader.read_bias_attributes_from_json('exp_data/sentiment/definitional_pairs.json')
    race_attr = data_loader.read_bias_attributes_from_json('exp_data/race/definitional_pairs.json')
    
    weats = {}
    for test in ['weat3', 'weat3b', 'weat4', 'weat5', 'weat5b', 'weat6', 'weat6b', 'weat7', 'weat7b', 'weat8', 'weat8b']:
        file = 'exp_data/seat/'+test+'.jsonl'
        print("read: "+file)
        weats.update({test: data_loader.read_dict_from_json(file)})
    weats.update({'angry_black_woman': data_loader.read_dict_from_json('exp_data/seat/angry_black_woman_stereotype.jsonl')})
    weats.update({'angry_black_woman2': data_loader.read_dict_from_json('exp_data/seat/angry_black_woman_stereotype_b.jsonl')})
    weats.update({'double_bind_competent': data_loader.read_dict_from_json('exp_data/seat/heilman_double_bind_competent_one_word.jsonl')})
    weats.update({'double_bind_likable': data_loader.read_dict_from_json('exp_data/seat/heilman_double_bind_likable_one_word.jsonl')})
    
    # read sentence templates and replacement rules
    template_dict = data_loader.read_dict_from_json('exp_data/templates.json')
    rules_dict = data_loader.read_dict_from_json('exp_data/template_rules.json')
   
    sent_generator = SentGenerator()

    print("test sentences (target words):")
    test_sent_sentiment = sent_generator.create_sent_from_template_dict(sent_test, template_dict, rules_dict, keys=['adjectives', 'things_singular', 'things_no_pronoun'])
    test_sent_prof = sent_generator.insert_to_templates(prof, template_dict['people_singular'], template_dict['token'])
    test_sent_comp = sent_generator.insert_to_templates(comp, template_dict['things_no_pronoun'], template_dict['token'])
    
    print("bias attribute sentences:")
    btest_sent_gender = sent_generator.create_attr_sent_from_template_dict(list(zip(*gender_attr)), template_dict, rules_dict)
    #print(btest_sent_gender)
    btest_sent_religion = sent_generator.create_attr_sent_from_template_dict(list(zip(*religion_attr)), template_dict, rules_dict)
    btest_sent_sentiment = sent_generator.create_attr_sent_from_template_dict(list(zip(*sentiment_attr)), template_dict, rules_dict)
    btest_sent_race = sent_generator.create_attr_sent_from_template_dict(list(zip(*race_attr)), template_dict, rules_dict)
    
    
    # create sentences for SEATs
    def create_seat_sent(weat_dict, template_dict, rules_dict):
        seat_dict = copy.deepcopy(weat_dict)
        for test in weat_dict.keys():
            for k in weat_dict[test].keys():
                category = weat_dict[test][k]['category']
                if 'Names' in category or category in ['Male', 'Female']:
                    seat_dict[test][k]['examples'] = sent_generator.insert_to_templates(weat_dict[test][k]['examples'], template_dict['names'], template_dict['token'])
                elif category in ['EuropeanAmericanTerms', 'AfricanAmericanTerms']:
                    seat_dict[test][k]['examples'] = sent_generator.insert_to_templates(weat_dict[test][k]['examples'], template_dict['people_description'], template_dict['token'])
                else:
                    seat_dict[test][k]['examples'] = sent_generator.create_sent_from_template_dict(weat_dict[test][k]['examples'], template_dict, rules_dict)
        return seat_dict
    
    seats = create_seat_sent(weats, template_dict, rules_dict)
    
    def embed_sent(embed_function):
        print("embed gender...")
        emb_gender = [embed_function(tup) for tup in btest_sent_gender]
        print("embed religion...")
        emb_religion = [embed_function(tup) for tup in btest_sent_religion]
        print("embed race...")
        emb_race = [embed_function(tup) for tup in btest_sent_race]
        print("embed sentiment...")
        emb_sentiment_tup = [embed_function(tup) for tup in btest_sent_sentiment]
    
        for k, v in seats.items():
            print("embed "+k+"...")
            for v2 in v.values():
                v2['embeddings'] = embed_function(v2['examples'])
        
        return {'gender': emb_gender, 'religion': emb_religion, 'race': emb_race, 'sentiment': emb_sentiment_tup}
    
    def cossim(x, y):
        return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))
    
    def mac(T, As):
        return np.mean([1-cossim(t,a) for t in T for A in As for a in A])
    
    def normalize(vectors: np.ndarray):
        norms = np.apply_along_axis(LA.norm, 1, vectors)
        vectors = vectors / norms[:, np.newaxis]
        return np.asarray(vectors)
    
    
    def compare_metrics(seats,log_file): # embeddings
        f = open(log_file,"w")
        print("opened file", log_file)
        f.write("test;eff_weat;pval_weat;eff_own;own_bias_mean;own_std;mac;cluster\n")
        result_dict = {}
        for test in seats.keys():
            if not ('embeddings' in seats[test]['attr1'].keys() and 'embeddings' in seats[test]['targ1'].keys() and 'embeddings' in seats[test]['attr2'].keys() and 'embeddings' in seats[test]['targ2'].keys()):
                print("skip "+test)
                continue
                 
                
            # normalize embeddings
            seats[test]['attr1']['embeddings'] = normalize(seats[test]['attr1']['embeddings'])
            seats[test]['attr2']['embeddings'] = normalize(seats[test]['attr2']['embeddings'])
            seats[test]['targ1']['embeddings'] = normalize(seats[test]['targ1']['embeddings'])
            seats[test]['targ2']['embeddings'] = normalize(seats[test]['targ2']['embeddings'])
    
            print(test)
    
            pval, esize = weat.run_tests(seats[test])
            
            esize2, bias_mean, bias_std = weat.run_tests_own(seats[test])
    
            mac_score = mac(seats[test]['targ1']['embeddings']+seats[test]['targ2']['embeddings'], [seats[test]['attr1']['embeddings'],seats[test]['attr2']['embeddings']])
    
            X_bef = np.vstack([seats[test]['targ1']['embeddings'],seats[test]['targ2']['embeddings']])
            y = np.asarray([0]*len(seats[test]['targ1']['embeddings'])+[1]*len(seats[test]['targ2']['embeddings']))
          #  cluster_score = distance_tests.cluster_test(X_bef, y, num=2)
            
            f.write(test+"&"+str(esize)+"&"+str(pval)+";&"+str(esize2)+"&"+str(bias_mean)+"&"+str(bias_std)+"&"+str(mac_score)+"\\"+"\n")
            
            test_list = []
            test_list.append(esize)
            test_list.append(esize2)
            test_list.append(mac_score)
            test_list.append(pval)
        
            
            result_dict[test] = test_list
            # get single word biases
            #W = np.vstack([weats[test]['targ1']['embeddings'],weats[test]['targ2']['embeddings']])
            #word_biases = [weat.own_metric.bias_w(w, A) for w in W]
    
        # TODO run other tests
            
        f.close()
        
        return result_dict
        
        
        
    weat_metrics = WEATMetrics()
    own_metrics = ImprovedBiasMetrics()
    weat = WEAT(weat_metrics, own_metrics)


    emb = embed_sent(model.embed)
    result_dict = compare_metrics(seats,savepath)
    
    
    # 3-5b european american vs african american pleasant vs unpleasant
    # 6-8b  gender
    # angry black woman racial
    # double bind gender
    return result_dict


