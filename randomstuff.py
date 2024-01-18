from scipy.special import softmax

classes = [
'Abyssinian'
 ,'American Bobtail'
 ,'American Curl'
 ,'American Shorthair'
 ,'Bengal'
 ,'Birman'
 ,'Bombay'
 ,'British Shorthair'
 ,'Egyptian Mau'
 ,'Exotic Shorthair'
 ,'Maine Coon'
 ,'Manx'
 ,'Norwegian Forest'
 ,'Persian'
 ,'Ragdoll'
 ,'Russian Blue'
 ,'Scottish Fold'
 ,'Siamese'
 ,'Sphynx'
 ,'Turkish Angora']

resultdict= {'Abyssinian': -9.62617301940918, 'American Bobtail': -12.792186737060547, 'American Curl': -5.849104881286621, 'American Shorthair': -14.968454360961914, 'Bengal': -8.734613418579102, 'Birman': -1.0677375793457031, 'Bombay': -20.60688591003418, 'British Shorthair': -4.479479789733887, 'Egyptian Mau': -18.375415802001953, 'Exotic Shorthair': -16.32208251953125, 'Maine Coon': -3.4632279872894287, 'Manx': -4.7236738204956055, 'Norwegian Forest': -11.613703727722168, 'Persian': -12.862920761108398, 'Ragdoll': 4.787437438964844, 'Russian Blue': -9.887436866760254, 'Scottish Fold': -16.151731491088867, 'Siamese': -6.416225910186768, 'Sphynx': -20.532411575317383, 'Turkish Angora': -19.605844497680664}

probabilities = softmax(list(resultdict.values()))

yep=dict(zip(classes, probabilities))

print(yep)
