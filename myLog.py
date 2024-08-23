def log(the_str, log_name, model_save=False):
    if model_save:
        the_str = 'Model Saved!!!' + the_str
    try:
        print(the_str)
        with open(log_name, 'a+', encoding='utf-8') as f:
            f.write(the_str+'\n')
    except:
        print('write error!!')