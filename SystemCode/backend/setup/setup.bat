REM MS-DOS cmd version

set ~=%userprofile%
set dateIngestFolder=2019-12-27T08;03;51.498312



zipline ingest -b quantopian-quandl

REM Custom alpaca bundle of ETF data
cp dotzipline.tgz %~%/.zipline/
tar -xvzf %~%/.zipline/dotzipline.tgz -C %~%/.zipline/

REM 
mkdir "%~%/.zipline/data/alpaca/2018-06-11T20;08;42.452595/minute_equities.bcolz"
cp "%~%/.zipline/data/quantopian-quandl/%dateIngestFolder%/minute_equities.bcolz/metadata.json" "%~%/.zipline/data/alpaca/2018-06-11T20;08;42.452595/minute_equities.bcolz/"
cp "%~%/.zipline/data/quantopian-quandl/%dateIngestFolder%/adjustments.sqlite" "%~%/.zipline/data/alpaca/2018-06-11T20;08;42.452595/"


REM cp `find %~%/.zipline/ -name "metadata.json"` %~%/.zipline/data/alpaca/2018-06-11T20\;08\;42.452595/minute_equities.bcolz/
REM cp `find %~%/.zipline/data/quantopian-quandl -name "adjustments.sqlite"` %~%/.zipline/data/alpaca/2018-06-11T20\;08\;42.452595/
