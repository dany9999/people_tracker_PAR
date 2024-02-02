

from collections import Counter


def tuple_con_valori_piu_frequenti(lista_di_tuple):
    # Inizializza un dizionario per tenere traccia delle frequenze di ciascun elemento
    frequenze = {'gender': Counter(), 'hat': Counter(), 'bag': Counter(),
                 'color_up': Counter(), 'color_low': Counter()}

    # Itera attraverso la lista di tuple e aggiorna le frequenze
    for tupla in lista_di_tuple:
        
        frequenze['gender'][tupla[0]] += 1
        frequenze['hat'][tupla[1]] += 1
        frequenze['bag'][tupla[2]] += 1
        frequenze['color_up'][tupla[3]] += 1
        frequenze['color_low'][tupla[4]] += 1

    # Inizializza una tupla con i valori più frequenti
    tupla_valori_piu_frequenti = (
        frequenze['gender'].most_common(1)[0][0],
        frequenze['hat'].most_common(1)[0][0],
        frequenze['bag'].most_common(1)[0][0],
        frequenze['color_up'].most_common(1)[0][0],
        frequenze['color_low'].most_common(1)[0][0]
    )

    return tupla_valori_piu_frequenti

#Esempio di utilizzo
lista_di_tuple = [(True, False, True, 'red', 'blue'),
                  (False, True, True, 'green', 'red'),
                  (True, False, False, 'red', 'blue'),
                  # Aggiungi altre tuple secondo necessità
                 ]

risultato = tuple_con_valori_piu_frequenti(lista_di_tuple)
print(risultato)

