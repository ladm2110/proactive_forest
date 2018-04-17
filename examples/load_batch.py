from examples import load_data


def get_batch_1():
    data = [
        ('balance_scale', load_data.load_balance_scale()),
        ('car', load_data.load_car()),
        ('cmc', load_data.load_cmc()),
        ('credit-g', load_data.load_credit()),
        ('diabetes', load_data.load_diabetes())
    ]
    return data


def get_batch_2():
    data = [
        ('ecoli', load_data.load_ecoli()),
        ('flags', load_data.load_flags()),
        ('glass', load_data.load_glass()),
        ('haberman', load_data.load_haberman()),
        ('heart-statlog', load_data.load_heart_statlog())
    ]
    return data


def get_batch_3():
    data = [
        ('ionosphere', load_data.load_ionosphere()),
        ('iris', load_data.load_iris()),
        ('kr-vs-kp', load_data.load_kr_vs_kp()),
        ('letter', load_data.load_letter()),
        ('liver', load_data.load_liver_disorder())
    ]
    return data


def get_batch_4():
    data = [
        ('lymph', load_data.load_lymph()),
        ('molecular', load_data.load_molecular()),
        ('nursery', load_data.load_nursery()),
        ('optdigits', load_data.load_optdigits()),
        ('page_blocks', load_data.load_page_blocks())
    ]
    return data


def get_batch_5():
    data = [
        ('pendigits', load_data.load_pendigits()),
        ('segment', load_data.load_segment()),
        ('solar_flare1', load_data.load_solar_flare1()),
        ('solar_flare2', load_data.load_solar_flare2()),
        ('sonar', load_data.load_sonar())
    ]
    return data


def get_batch_6():
    data = [
        ('spambase', load_data.load_spambase()),
        ('splice', load_data.load_splice()),
        ('tae', load_data.load_tae()),
        ('vehicle', load_data.load_vehicle()),
        ('vowel', load_data.load_vowel())
    ]
    return data


def get_batch_7():
    data = [
        ('wdbc', load_data.load_wdbc()),
        ('wine', load_data.load_wine())
    ]
    return data


def get_batch_8():
    data = [
        ('mfeat_factors', load_data.load_mfeat_factors()),
        ('mfeat_karhunene', load_data.load_mfeat_karhunen()),
        ('mfeat_morphological', load_data.load_morphological()),
        ('mfeat_pixel', load_data.load_pixel()),
        ('mfeat_zernike', load_data.load_zernike()),
        ('mfeat_fourier', load_data.load_fourier())
    ]
    return data


def get_all():
    data = [
        ('balance_scale', load_data.load_balance_scale()),
        ('car', load_data.load_car()),
        ('cmc', load_data.load_cmc()),
        ('credit-g', load_data.load_credit()),
        ('diabetes', load_data.load_diabetes()),
        ('ecoli', load_data.load_ecoli()),
        ('flags', load_data.load_flags()),
        ('glass', load_data.load_glass()),
        ('haberman', load_data.load_haberman()),
        ('heart-statlog', load_data.load_heart_statlog()),
        ('ionosphere', load_data.load_ionosphere()),
        ('iris', load_data.load_iris()),
        ('kr-vs-kp', load_data.load_kr_vs_kp()),
        ('letter', load_data.load_letter()),
        ('liver', load_data.load_liver_disorder()),
        ('lymph', load_data.load_lymph()),
        ('molecular', load_data.load_molecular()),
        ('nursery', load_data.load_nursery()),
        ('optdigits', load_data.load_optdigits()),
        ('page_blocks', load_data.load_page_blocks()),
        ('pendigits', load_data.load_pendigits()),
        ('segment', load_data.load_segment()),
        ('solar_flare1', load_data.load_solar_flare1()),
        ('solar_flare2', load_data.load_solar_flare2()),
        ('sonar', load_data.load_sonar()),
        ('spambase', load_data.load_spambase()),
        ('splice', load_data.load_splice()),
        ('tae', load_data.load_tae()),
        ('vehicle', load_data.load_vehicle()),
        ('vowel', load_data.load_vowel()),
        ('wdbc', load_data.load_wdbc()),
        ('wine', load_data.load_wine())
    ]
    return data
