Ergebnisse der Analyse:

Die erste Implementierung habe ich mit Hilfe von ChatGPT in Python umgesetzt. 
Erst findet die schnelle Fourieranalyse statt. Im nächsten Schritt werden die Ergebnisse geplottet. Einmal ein Spektogramm. Dort zu sehen ist: Die Zeit auf der x-Achse und die Frequenz auf der y-Achse. Die Farbe zeigt die Magnitude an.
Die Fourieranalyse habe ich einmal mit einer Blockgröße von 512 und einmal mit einer Blockgröße von 256 ausgeführt und jeweils einem Shift von 1.
Bei einer Blockgröße von 1024 wurde die Analyse noch ausgeführt, aber das Plotten des Ergebnisses hat nicht mehr funktioniert.

Für die anderen Teile der Aufgabe habe ich den Speicherverbrauch gemessen. Einmal in Python, einmal in Java und dann nochmal in Python, aber auf der WSL, also einem anderen Betriebssystem.

Mit tracemalog habe ich den Speicherbedarf der Analyse in Python verfolgt, indem ich ich die Messung des Speicherplatzes vor der Schleife, die die Analyse ausführt, starte und nach der Schleife wieder schließe.
Für Windows habe ich den Speicherbedarf auch graphisch dargestellt, auf Linux war das ohne Weiteres nicht machbar, weswegen ich den Speicherverbrauch in der Konsole ausgegeben habe.
In Java habe ich den Speicherverbrauch mit memoryusage.used verfolgt, was den Speicherverbrauch der JVM zu einem bestimmten Zeitpunkt anzeigt. Die Werte habe ich gespeichert und geplottet.


Zwischen den Betriebssystemen gab es einen Unterschied beim Umgang eines zu hohem Verbrauch des Arbeitspeichers. Beim Ausführen in Windows hat sich das Programm bei einem zu hohen Speicherverbrauch auf gehangen. Soweit, dass sogar mein ganzer Laptop langsamer wurde. Außerdem ist aufgefallen, dass der Speicherbedarf bei jeder Ausführung identisch ist (bei gleichem OS). Zwischen Linux und Windows gibt es einen Unterschied von 2MB.
In Linux wurde der Prozess gekillt, wenn der Speicherverbrauch zu groß wurde. 
Zwischen Python und Java gab es einen Unterschied zwischen der Speichervewaltung selbst. In Python ist der Verbrauch sehr linear gestiegen. 
In Java hingegen wurde in regelmäßigen Abständen Speicher freigegeben, sodass der Höchstverbrauch sich in Grenzen gehalten hat. Ich vermute, dass der Garbagecollector dafür zuständig war. Außerdem musste ich in Java den Speicherplatz der JVM vergrößern, um mein Programm ausführen zu können.