import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_balance_scale():
    dataset = pd.read_csv('../data/balance-scale.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_car():
    dataset = pd.read_csv('../data/car.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_cmc():
    dataset = pd.read_csv('../data/cmc.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_credit():
    dataset = pd.read_csv('../data/credit-g.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_diabetes():
    dataset = pd.read_csv('../data/diabetes.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_ecoli():
    dataset = pd.read_csv('../data/ecoli.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_flags():
    dataset = pd.read_csv('../data/flags_religion.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_glass():
    dataset = pd.read_csv('../data/glass.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['Type']
    X = dataset.drop('Type', axis=1)
    return X, y


def load_haberman():
    dataset = pd.read_csv('../data/haberman.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_heart_statlog():
    dataset = pd.read_csv('../data/heart-statlog.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_ionosphere():
    dataset = pd.read_csv('../data/ionosphere.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_iris():
    dataset = pd.read_csv('../data/iris.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_kr_vs_kp():
    dataset = pd.read_csv('../data/kr-vs-kp.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_letter():
    dataset = pd.read_csv('../data/letter.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_liver_disorder():
    dataset = pd.read_csv('../data/liver-disorders.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['selector']
    X = dataset.drop('selector', axis=1)
    return X, y


def load_lymph():
    dataset = pd.read_csv('../data/lymph.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_molecular():
    dataset = pd.read_csv('../data/molecular-biology_promoters.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_nursery():
    dataset = pd.read_csv('../data/nursery.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_optdigits():
    dataset = pd.read_csv('../data/optdigits.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_page_blocks():
    dataset = pd.read_csv('../data/page-blocks.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_pendigits():
    dataset = pd.read_csv('../data/pendigits.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_segment():
    dataset = pd.read_csv('../data/segment.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_solar_flare1():
    dataset = pd.read_csv('../data/solar-flare_1.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_solar_flare2():
    dataset = pd.read_csv('../data/solar-flare_2.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_sonar():
    dataset = pd.read_csv('../data/sonar.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['Class']
    X = dataset.drop('Class', axis=1)
    return X, y


def load_spambase():
    dataset = pd.read_csv('../data/spambase.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_splice():
    dataset = pd.read_csv('../data/splice.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['Class']
    X = dataset.drop('Class', axis=1)
    return X, y


def load_tae():
    dataset = pd.read_csv('../data/tae.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_vehicle():
    dataset = pd.read_csv('../data/vehicle.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['Class']
    X = dataset.drop('Class', axis=1)
    return X, y


def load_vowel():
    dataset = pd.read_csv('../data/vowel.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['Class']
    X = dataset.drop('Train or Test', axis=1)
    X = X.drop('Class', axis=1)
    return X, y


def load_wdbc():
    dataset = pd.read_csv('../data/wdbc.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_wine():
    dataset = pd.read_csv('../data/wine.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    return X, y


def load_mfeat_factors():
    dataset = pd.read_csv('../data/mfeat-factors.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y


def load_mfeat_karhunen():
    dataset = pd.read_csv('../data/mfeat-karhunen.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y


def load_mfeat_morphological():
    dataset = pd.read_csv('../data/mfeat-morphological.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y


def load_mfeat_pixel():
    dataset = pd.read_csv('../data/mfeat-pixel.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y


def load_mfeat_zernike():
    dataset = pd.read_csv('../data/mfeat-zernike.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y


def load_mfeat_fourier():
    dataset = pd.read_csv('../data/mfeat-fourier.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset['class']
    X = dataset.drop('class', axis=1)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y))
    return X, y
