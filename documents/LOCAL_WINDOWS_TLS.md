# Lokaler HTTPS-Abruf unter Windows und Python 3.14

In der verwendeten Python-3.14-Umgebung kann `requests` beim TLS-Handshake
mit `OPENSSL_Uplink(...): no OPENSSL_Applink` abbrechen. Das ist eine
Inkompatibilitaet der lokalen OpenSSL-Bindung, kein fehlendes Projektzertifikat.

Die Datenclients verwenden deshalb unter Windows `curl.exe`. Dieses nutzt den
Windows-Zertifikatsspeicher und laesst die TLS-Zertifikatspruefung aktiv.
Auf anderen Betriebssystemen verwenden die Clients weiterhin `requests`.
Betroffen sein können sowohl SMARD- als auch Open-Meteo-Abrufe.

Es ist keine eigene CA zu erzeugen oder zu installieren. Insbesondere darf
`verify=False` nicht als Umgehung eingesetzt werden.

Bei einem solchen Fehler zuerst `curl.exe --version` prüfen und anschließend
den Abruf erneut ausführen. Der Fallback ist Teil der Clients; für die
Projektaufrufe ist keine zusätzliche Konfiguration erforderlich.
