# Obiettivo

L'obiettivo di questo progetto è realizzare un modello che a partire dai dati relativi al volto di un osservatore di un'opera d'arte possa predire i suoi livelli di arousal, valence, likability, rewatch e familiarity.

In particolare, l'arousal e la valence indicano rispettivamente lo stato di eccitazione che si manifesta in risposta ad uno stimolo e quanto questo sia positivo o negativo. La likability indica quanto uno stimolo sia piaciuto. Con la rewatch si classifica il desiderio di rivedere un certo stimolo. La familiarity indica se input era già noto allo spettatore.

# Dati

Il dataset è stato ottenuto sfruttando un sito realizzato ad hoc ed una piattaforma di crowdsourcing, Prolific, per la ricerca dei partecipanti.

Durante la sperimentazione si è mostrato ad ogni utente un totale di 12 opere d'arte. Le reazioni dell'utente venivano registrate attraverso la sua webcam e dopo ogni stimolo veniva richiesto di indicare esplicitamente i feedback relativi ai parametri appena descitti. Di conseguenza, per ogni partecipante sono stati ottenuti 12 video e 5 feedback espliciti per video più i dati relativi ad un questionario finale.

Alla fine della spetimentazione si sono ottenuti i dati relativi a 111 partecipanti.

# Modello Deep

In questa repository sono contenuti i codici relativi a dei modelli basati su LSTM+CNN, con CNN preaddrestrate. La CNN permette di lavorare sul dato spaziale, la LSTM ha il compito di dare significato temporale al dato.

In particolare, nel file model della cartella model sono codificate le strutture delle reti utilizzate durante il training.

__VisitorNetInception:__ Utilizza come backbone una la nota CNN InceptionNet, preaddrestrata su vggface2, un dataset per il riconoscimento dei volti. Come consigliato in vari testi, sono stati freezati i layer più bassi della CNN. I dati estratti dalla CNN sono passati alla rete LSTM, con un solo layer, e successivamente ci sono due strati lineari. Prima di ogni strato lineare è stato inserito un layer di dropout per evitare il fenomeno dell'overfitting. In più è stata inserita una leakyRelu prima dello strato fully connected finale per evitare la problematica del vanishing gradient.

__VisitorNetEmonet:__ Come la precedente ma utilizza come backcone Emonet, estratta e adattata dal paper "Estimation of continuous valence and arousal levels from faces in naturalistic conditions".

__VisitorNetResnet:__ La rete più semplice tra le tre. Utilizza come backbone resnet34, preaddrestrata su Imagenet, senza alcun layer freezato. A differenza delle precedenti, solamente tra gli ultimi due layer fully connected è presente una relu e un dropout.

## Preparazione dei Dati

Per poter feedare il video alla rete è necessario effettuare un preprocessing dedi video in cui dai ognuno si estrae un immagine per ogni frame. Successivamente, è stata individuata una libreria, Video-Dataset-Loading, che potesse semplificare il sampling dei frame e la costruzione del data loader per pytorch.

Ogni video viene suddiviso in 3 parti e per ogni parte vengono scelti 12 frame.

Successivamente, il dataset è stato suddiviso in training, validation e test.

## Training

Il training è stato eseguito testando diversi iperparametri della rete. In particolare, sono state provate tutti e 3 i modelli modificando batch size, learning rate, optimizer e scheduler.

## Risultati

Nella cartella saved/log sono presenti alcuni log dei test effettuati. La fluttuazione dei risultati ha reso necessario un'analisi più accurata di quanto succedesse nella rete. Infatti, è stato possibile verificare che da un training all'altro la rete avesse dei comportamenti anomali.

In particolare, si è notato che la rete aveva ottenuto dei buoni risultati sia sul train che sul validation ma l'accuracy e la loss sul test risultavano essere mediocri. Da un'analisi più approfondita, attraverso l'utilizzo di tensorboard, si è notata una importante fluttuazione di accuracy e loss tra un batch e l'altro. Questa fluttuazione ha di conseguenza causato risultati molto discrepanti tra un test e l'altro.

I risultati migliori sono stati ottenuti per la label rewatch in combinazione con la backbone InceptionNet con un'accuracy del 47% (log nella cartella saved/log/smaller).