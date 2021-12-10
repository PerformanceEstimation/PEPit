Create new example
==================

Google style docstring
----------------------

Pycharm setting:
^^^^^^^^^^^^^^^^
Pycharm/Preferences…/Tools/Python Integrated Tools/Docstrings/Docstring format -> Google

Table of content:
^^^^^^^^^^^^^^^^^
- Define Pb it solves (with f\_\star \\triangleq )

- Name algo in boldface

- Introduce perf metric (out < tau * in)

- Describe alg main step

- Write theoretical result (tau + Upper/Lower/Tight in italic bold)

- Refs (preferably arxiv and with arxiv version)

- Params (indicate optional or not)

- Returns

- Example

PEPit: Liste des trucs à verifier concernant les exemples:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Virer les hypotheses qui n’en sont pas comme « non-smooth ».

- Privilégier \equiv au lieu de = quand on minimise F\_\\star \\triangleq F(x) = f_1(x) + f_2(x).

- Mettre un peu plus de "verbose" avant les équations décrivant la méthode.

- Add arrow on gamma if sequence.

- Uniformiser les indices: k->t, n for final step.

- Only n steps, not n+1

- Utiliser eqnarray environment quand on enchaine plusieurs equations

- Aligner les examples sur les papiers respectifs (éviter les formulations différentes)

- Verifier les théorèmes (les taux sont ils tight? Upper? Lower? éventuellement preciser si on connait un tau expérimentalement qui est different du théorème connu.)

- Vérifier les références et citer les passages exacts, pas juste l article.

- Format des références (Initiales des prénoms + noms complets puis année entre parentheses, titre et enfin journal entre parentheses.) Ajouter Egalement une url pour rendre la reference clickable.

- Presentation des paramètres dans un bloc « Args ». De meme pour Return.

- Argument verbose=true explicitement dans Exemple

- Alignement entre exemple docstrings et if name==main (mettre exactement la meme formule)

- Dans le texte qui s’affiche pour comparer theoretical tau et PEPit-tau, verifier qu’il y ait bien 2  « \t » vs 1 seule pour aligner les deux résultats.

- Enlever les espaces dans les normes (entre le symbole norme et le vecteur concerné)

- Vérifier que l'exemple est bien dans les tests

- stepsize, step size -> step-size

- algorithm -> method

- fast -> accelerated
