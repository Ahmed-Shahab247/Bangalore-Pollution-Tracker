import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import PropTypes from 'prop-types';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import {
  Activity,
  ArrowUpRight,
  Check,
  Leaf,
  MapPin,
  Navigation,
  Plus,
  RefreshCw,
  ShieldAlert,
  Wind,
  X,
} from 'lucide-react';
import './App.css';

function getAirQuality(pm25) {
  if (!Number.isFinite(pm25)) return { label: 'Unavailable', tone: 'unknown' };
  if (pm25 <= 30) return { label: 'Good', tone: 'good' };
  if (pm25 <= 60) return { label: 'Satisfactory', tone: 'satisfactory' };
  if (pm25 <= 90) return { label: 'Moderate', tone: 'moderate' };
  if (pm25 <= 120) return { label: 'Poor', tone: 'poor' };
  if (pm25 <= 250) return { label: 'Very poor', tone: 'very-poor' };
  return { label: 'Severe', tone: 'severe' };
}

function getIndicativeAqiQuality(aqi) {
  if (!Number.isFinite(aqi)) return { label: 'Unavailable', tone: 'unknown' };
  if (aqi <= 50) return { label: 'Good', tone: 'good' };
  if (aqi <= 100) return { label: 'Satisfactory', tone: 'satisfactory' };
  if (aqi <= 200) return { label: 'Moderate', tone: 'moderate' };
  if (aqi <= 300) return { label: 'Poor', tone: 'poor' };
  if (aqi <= 400) return { label: 'Very poor', tone: 'very-poor' };
  return { label: 'Severe', tone: 'severe' };
}

function getRegionMetric(region, mapMetric) {
  return mapMetric === 'aqi' ? region.indicative_aqi : region.pm25;
}

function getRegionQuality(region, mapMetric) {
  return mapMetric === 'aqi'
    ? getIndicativeAqiQuality(region.indicative_aqi)
    : getAirQuality(region.pm25);
}

function getDistanceKm(lat1, lon1, lat2, lon2) {
  const radians = (degrees) => (degrees * Math.PI) / 180;
  const deltaLat = radians(lat2 - lat1);
  const deltaLon = radians(lon2 - lon1);
  const haversine = (
    Math.sin(deltaLat / 2) ** 2
    + Math.cos(radians(lat1))
      * Math.cos(radians(lat2))
      * Math.sin(deltaLon / 2) ** 2
  );
  return 6371 * 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
}

function getReadingAgeLabel(observedAt) {
  if (!observedAt) return '';
  const ageHours = Math.max(0, Math.floor((Date.now() - new Date(observedAt).getTime()) / 3600000));
  return ageHours < 1 ? ' · updated within the last hour' : ` · ${ageHours}h old`;
}

function createAirQualityIcon(region, mapMetric) {
  const value = getRegionMetric(region, mapMetric);
  const tone = getRegionQuality(region, mapMetric).tone;
  const markerValue = mapMetric === 'aqi'
    ? Math.round(value)
    : value < 1 ? value.toFixed(2) : value.toFixed(1);
  return L.divIcon({
    className: 'aqi-marker-icon',
    html: `<span class="aqi-marker ${tone}">${markerValue}</span>`,
    iconSize: [42, 34],
    iconAnchor: [21, 17],
    popupAnchor: [0, -18],
  });
}

function MapResizeHandler() {
  const map = useMap();

  useEffect(() => {
    const container = map.getContainer().parentElement;
    let frameId;
    const refreshMapSize = () => {
      cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(() => {
        map.invalidateSize({ pan: false, debounceMoveend: true });
      });
    };

    refreshMapSize();
    const observer = new ResizeObserver(refreshMapSize);
    observer.observe(container);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(frameId);
    };
  }, [map]);

  return null;
}

function FitRegions({ regions, userLocation }) {
  const map = useMap();

  useEffect(() => {
    if (regions.length === 0 || userLocation) return;

    const bounds = L.latLngBounds(regions.map((region) => [region.lat, region.lon]));
    map.fitBounds(bounds, { padding: [28, 28], maxZoom: 11 });
  }, [map, regions, userLocation]);

  return null;
}

function FocusNearestRegion({ result }) {
  const map = useMap();

  useEffect(() => {
    if (!result) return;
    map.flyTo([result.region.lat, result.region.lon], 13, { duration: 0.8 });
  }, [map, result]);

  return null;
}

function FocusUserLocation({ location }) {
  const map = useMap();

  useEffect(() => {
    if (!location) return;
    map.flyTo([location.lat, location.lon], 13, { duration: 0.8 });
  }, [location, map]);

  return null;
}

FitRegions.propTypes = {
  regions: PropTypes.arrayOf(PropTypes.shape({
    lat: PropTypes.number.isRequired,
    lon: PropTypes.number.isRequired,
  })).isRequired,
  userLocation: PropTypes.shape({
    lat: PropTypes.number.isRequired,
    lon: PropTypes.number.isRequired,
  }),
};

FocusNearestRegion.propTypes = {
  result: PropTypes.shape({
    region: PropTypes.shape({
      lat: PropTypes.number.isRequired,
      lon: PropTypes.number.isRequired,
    }).isRequired,
  }),
};

FocusUserLocation.propTypes = {
  location: PropTypes.shape({
    lat: PropTypes.number.isRequired,
    lon: PropTypes.number.isRequired,
  }),
};

export default function App() {
  const [regions, setRegions] = useState([]);
  const [knownAreas, setKnownAreas] = useState([]);
  const [regionsError, setRegionsError] = useState('');
  const [areasError, setAreasError] = useState('');
  const [regionsLoading, setRegionsLoading] = useState(true);
  const [watchlist, setWatchlist] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  const [loadingGeo, setLoadingGeo] = useState(false);
  const [geoError, setGeoError] = useState('');
  const [search, setSearch] = useState('');
  const [nearestResult, setNearestResult] = useState(null);
  const [nearestError, setNearestError] = useState('');
  const [lookingUpNearest, setLookingUpNearest] = useState(false);
  const [mapMetric, setMapMetric] = useState('pm25');
  const [forecast, setForecast] = useState(null);
  const [forecastStatus, setForecastStatus] = useState(null);
  const [forecastError, setForecastError] = useState('');
  const [forecastLoading, setForecastLoading] = useState(true);
  const [forecastRefreshCount, setForecastRefreshCount] = useState(0);
  const [forecastHorizon, setForecastHorizon] = useState(3);

  useEffect(() => {
    axios.get('http://localhost:8000/api/areas')
      .then((response) => {
        setKnownAreas(response.data);
        setAreasError('');
      })
      .catch((error) => {
        console.error('Error fetching area directory:', error);
        setAreasError(error.response?.data?.detail ?? 'Could not load the area directory.');
      });

    axios.get('http://localhost:8000/api/regions')
      .then((response) => {
        setRegions(response.data);
        setRegionsError('');
      })
      .catch((error) => {
        console.error('Error fetching regions:', error);
        setRegionsError(error.response?.data?.detail ?? 'Could not load live station data.');
      })
      .finally(() => setRegionsLoading(false));
  }, []);

  useEffect(() => {
    let active = true;
    axios.get('http://localhost:8000/api/forecast/status')
      .then((response) => {
        if (active) setForecastStatus(response.data);
      })
      .catch(() => {
        if (active) {
          setForecastStatus({
            available: false,
            message: 'Forecast status could not be checked. Make sure the backend is running, then reload.',
          });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!forecastStatus) return undefined;
    if (!forecastStatus.available) {
      setForecast(null);
      setForecastLoading(false);
      return undefined;
    }

    let active = true;
    setForecastLoading(true);
    axios.get('http://localhost:8000/api/forecast', {
      params: {
        horizon_hours: forecastHorizon,
        ...(forecastRefreshCount > 0 ? { force_refresh: true } : {}),
      },
    })
      .then((response) => {
        if (!active) return;
        setForecast(response.data);
        setForecastError('');
      })
      .catch((error) => {
        if (!active) return;
        console.error('Error fetching PM2.5 forecast:', error);
        setForecastError(error.response?.data?.detail ?? 'Could not load the PM2.5 forecast.');
      })
      .finally(() => {
        if (active) setForecastLoading(false);
      });

    return () => {
      active = false;
    };
  }, [forecastHorizon, forecastRefreshCount, forecastStatus]);

  const handleGetLocation = () => {
    setGeoError('');
    if (!navigator.geolocation) {
      setGeoError('Location services are not supported by this browser.');
      return;
    }

    setLoadingGeo(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setUserLocation({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          name: 'Your current location',
        });
        setLoadingGeo(false);
      },
      (error) => {
        console.error('Geolocation error:', error);
        setGeoError(
          error.code === 1
            ? 'Location access is blocked. Allow location permission for this site in your browser, then try again.'
            : error.code === 2
              ? 'Your device could not determine its location. Check that location services are enabled.'
              : 'Location lookup timed out. Check your connection and try again.',
        );
        setLoadingGeo(false);
      },
      { enableHighAccuracy: true, timeout: 10000 },
    );
  };

  const addToWatchlist = (region) => {
    setWatchlist((current) => (
      current.some((item) => item.name === region.name) ? current : [...current, region]
    ));
  };

  const removeFromWatchlist = (name) => {
    setWatchlist((current) => current.filter((item) => item.name !== name));
  };

  const namedAreas = regions.filter((region) => region.is_area);
  const listedAreas = namedAreas.length > 0
    ? namedAreas
    : knownAreas.map((area) => ({
      ...area,
      id: `area-${area.name.toLowerCase().replaceAll(' ', '-')}`,
      is_area: true,
    }));
  const nearestUserArea = userLocation && namedAreas.length
    ? namedAreas.reduce((nearest, area) => {
      const distance = getDistanceKm(
        userLocation.lat,
        userLocation.lon,
        area.lat,
        area.lon,
      );
      return !nearest || distance < nearest.distance
        ? { region: area, distance }
        : nearest;
    }, null)
    : userLocation && listedAreas.length
      ? listedAreas.reduce((nearest, area) => {
        const distance = getDistanceKm(
          userLocation.lat,
          userLocation.lon,
          area.lat,
          area.lon,
        );
        return !nearest || distance < nearest.distance
          ? { region: area, distance }
          : nearest;
      }, null)
    : null;
  const matchingRegions = listedAreas.filter((region) => (
    region.name.toLowerCase().includes(search.trim().toLowerCase())
  ));
  const filteredRegions = nearestResult ? [nearestResult.region] : matchingRegions;
  const areaReadings = regions.filter(
    (region) => region.is_area && Number.isFinite(region.pm25),
  );
  const averagePm25 = areaReadings.length
    ? Math.round(areaReadings.reduce((total, region) => total + region.pm25, 0) / areaReadings.length)
    : null;
  const averageQuality = averagePm25 === null ? null : getAirQuality(averagePm25);
  const monitoredCount = areaReadings.length;
  const regionIcons = useMemo(
    () => new Map(
      regions
        .filter((region) => Number.isFinite(getRegionMetric(region, mapMetric)))
        .map((region) => [
          region.id ?? region.name,
          createAirQualityIcon(region, mapMetric),
        ]),
    ),
    [mapMetric, regions],
  );
  const mapRegions = regions.filter(
    (region) => Number.isFinite(getRegionMetric(region, mapMetric))
      && (!region.is_area || region.is_interpolated),
  );

  const handleFindClosest = async (event) => {
    event.preventDefault();
    const query = search.trim();
    if (query.length < 2) return;

    setLookingUpNearest(true);
    setNearestError('');
    setNearestResult(null);
    try {
      const response = await axios.get('http://localhost:8000/api/regions/nearest', {
        params: { query },
      });
      setNearestResult(response.data);
    } catch (error) {
      console.error('Error finding nearest area:', error);
      setNearestError(
        error.response?.data?.detail ?? 'Could not look up that area. Please try again.',
      );
    } finally {
      setLookingUpNearest(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Bengaluru Airwatch home">
          <span className="brand-mark"><Activity size={21} strokeWidth={2.5} /></span>
          <span className="brand-name">air<span>watch</span></span>
        </a>
        <div className="topbar-right">
          <span className={`network-status ${regionsError ? 'network-status-error' : ''}`}>
            <span className="status-dot" />
            {regionsError ? 'Live data unavailable' : regionsLoading ? 'Connecting to stations' : 'Live station data'}
          </span>
          <button className="location-button" onClick={handleGetLocation} disabled={loadingGeo}>
            <Navigation size={16} />
            <span>{loadingGeo ? 'Finding you…' : 'Use my location'}</span>
          </button>
        </div>
      </header>

      <main className="dashboard">
        <section className="welcome-row">
          <div>
            <p className="eyebrow"><span className="eyebrow-line" /> CITY AIR, AT A GLANCE</p>
            <h1>A clearer view of <span>Bengaluru.</span></h1>
            <p className="welcome-copy">Explore neighborhood air quality and keep the places you care about close.</p>
          </div>
          <div className="date-chip"><span className="status-dot" /> Greater Bengaluru</div>
        </section>

        <section className="overview-grid" aria-label="Air quality overview">
          <article className="overview-card overview-card-primary">
            <div className="card-icon"><Wind size={19} /></div>
            <p className="metric-label">Average area PM2.5</p>
            <div className="metric-value">{averagePm25 ?? '—'} <span>µg/m³</span></div>
            <div className={`quality-pill ${averageQuality?.tone ?? ''}`}>
              <span className="quality-dot" />
              {averageQuality?.label ?? 'Waiting for data'}
            </div>
          </article>
          <article className="overview-card">
            <div className="card-icon card-icon-lime"><MapPin size={19} /></div>
            <p className="metric-label">Areas covered</p>
            <div className="metric-value">{(regionsError ? 0 : monitoredCount).toString().padStart(2, '0')} <span>areas</span></div>
            <p className="metric-note">Areas with a fresh nearby station reading</p>
          </article>
          <article className="overview-card">
            <div className="card-icon card-icon-blue"><Leaf size={19} /></div>
            <p className="metric-label">Your watchlist</p>
            <div className="metric-value">{watchlist.length.toString().padStart(2, '0')} <span>saved</span></div>
            <p className="metric-note">Neighborhoods you’re keeping an eye on</p>
          </article>
        </section>

        <section
          className="forecast-panel"
          aria-label={`${forecastHorizon}-hour PM2.5 forecast`}
        >
          <div className="forecast-heading">
            <div>
              <p className="section-kicker">EXPERIMENTAL CITY-LEVEL OUTLOOK</p>
              <h2>{forecastHorizon}-hour PM2.5 forecast</h2>
              <p>Weather-driven estimate trained against historical Bengaluru ground-station PM2.5 observations.</p>
            </div>
            <div className="forecast-actions">
              <div className="map-mode-toggle forecast-horizon-toggle" role="group" aria-label="Forecast horizon">
                {[3, 12].map((horizon) => (
                  <button
                    key={horizon}
                    type="button"
                    className={forecastHorizon === horizon ? 'is-selected' : ''}
                    aria-pressed={forecastHorizon === horizon}
                    disabled={!forecastStatus?.available || forecastLoading}
                    onClick={() => setForecastHorizon(horizon)}
                  >
                    {horizon} hours
                  </button>
                ))}
              </div>
              <button
                className="forecast-refresh"
                onClick={() => setForecastRefreshCount((count) => count + 1)}
                disabled={!forecastStatus?.available || forecastLoading}
                aria-label="Refresh PM2.5 forecast"
              >
                <RefreshCw size={14} />
                {forecastLoading ? 'Updating…' : 'Refresh'}
              </button>
            </div>
          </div>
          {forecastLoading && <p className="forecast-message">Loading forecast…</p>}
          {forecastStatus && !forecastStatus.available && (
            <p className="forecast-message forecast-unavailable" role="status">
              {forecastStatus.message}
            </p>
          )}
          {forecastError && <p className="forecast-message forecast-error" role="alert">{forecastError}</p>}
          {forecast && !forecastError && (
            <>
              <div className="forecast-results">
                <div className="forecast-primary">
                  <span>Predicted PM2.5</span>
                  <strong>{forecast.pm25_prediction} <small>µg/m³</small></strong>
                  <span className={`quality-pill ${getAirQuality(forecast.pm25_prediction).tone}`}>
                    <span className="quality-dot" />
                    {getAirQuality(forecast.pm25_prediction).label}
                  </span>
                </div>
                <div className="forecast-detail">
                  <span>Forecast for</span>
                  <strong>
                    {new Date(forecast.forecast_for).toLocaleString('en-IN', {
                      weekday: 'short',
                      hour: 'numeric',
                      minute: '2-digit',
                    })}
                  </strong>
                </div>
                <div className="forecast-detail">
                  <span>Approx. prediction range</span>
                  <strong>
                    {forecast.prediction_interval.low}–{forecast.prediction_interval.high} µg/m³
                  </strong>
                </div>
                <div className="forecast-detail">
                  <span>
                    Monthly holdout MAE ({forecast.selected_model?.replaceAll('_', ' ')}) · persistence · daily baseline
                  </span>
                  <strong>
                    {forecast.model_mae} · {forecast.persistence_mae} · {forecast.seasonal_mae} µg/m³
                  </strong>
                </div>
              </div>
              {['persistence', 'seasonal_24h'].includes(forecast.selected_model) && (
                <p className="forecast-warning">
                  No trained model beat the baselines in testing, so this forecast uses the best-performing baseline.
                </p>
              )}
              <p className="forecast-disclaimer">{forecast.data_note}</p>
            </>
          )}
        </section>

        {geoError && (
          <div className="inline-alert" role="alert">
            <ShieldAlert size={17} />
            <span>{geoError}</span>
            <button onClick={() => setGeoError('')} aria-label="Dismiss location message"><X size={16} /></button>
          </div>
        )}
        {regionsError && (
          <div className="inline-alert" role="alert">
            <ShieldAlert size={17} />
            <span>{regionsError}</span>
          </div>
        )}

        <section className="content-grid">
          <aside className="side-column">
            <article className="panel watchlist-panel">
              <div className="panel-heading">
                <div>
                  <p className="section-kicker">PERSONAL DASHBOARD</p>
                  <h2>Your watchlist</h2>
                </div>
                <span className="count-badge">{watchlist.length}</span>
              </div>
              {watchlist.length === 0 ? (
                <div className="empty-watchlist">
                  <span className="empty-icon"><MapPin size={20} /></span>
                  <p className="empty-title">Your places, all in one spot</p>
                  <p>Pin a neighborhood below or from the map to start tracking it.</p>
                </div>
              ) : (
                <div className="watchlist-items">
                  {watchlist.map((item) => {
                    const value = getRegionMetric(item, mapMetric);
                    const quality = getRegionQuality(item, mapMetric);
                    return (
                      <div className="watchlist-item" key={item.name}>
                        <span className={`place-marker ${quality.tone}`}><MapPin size={16} /></span>
                        <div className="place-info">
                          <strong>{item.name}</strong>
                          <span>
                            <i className={`tiny-dot ${quality.tone}`} />
                              {Number.isFinite(value)
                                ? mapMetric === 'aqi'
                                  ? `${quality.label} · Indicative AQI ${value}`
                                  : `${quality.label} · PM2.5 ${value}`
                                : mapMetric === 'aqi'
                                  ? 'Indicative AQI unavailable'
                                  : 'No fresh PM2.5 reading'}
                          </span>
                        </div>
                        <button
                          className="icon-button remove-button"
                          onClick={() => removeFromWatchlist(item.name)}
                          aria-label={`Remove ${item.name} from watchlist`}
                        >
                          <X size={16} />
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </article>

            {userLocation && (
              <article className="location-card">
                <div className="location-card-icon"><Navigation size={17} /></div>
                <div>
                  <p className="location-title">You’re on the map</p>
                  {nearestUserArea ? (
                    <>
                      <p className="location-coordinates">
                        Nearest listed area: {nearestUserArea.region.name} · {nearestUserArea.distance.toFixed(1)} km
                        {getReadingAgeLabel(nearestUserArea.region.observed_at)}
                      </p>
                      <p className="location-quality">
                        {regionsError
                          ? 'Live pollution score unavailable while station data is offline'
                          : mapMetric === 'aqi'
                          ? Number.isFinite(nearestUserArea.region.indicative_aqi)
                            ? <>Indicative AQI <b>{nearestUserArea.region.indicative_aqi}</b> · {getIndicativeAqiQuality(nearestUserArea.region.indicative_aqi).label}</>
                            : 'Indicative AQI unavailable for this area'
                          : Number.isFinite(nearestUserArea.region.pm25)
                            ? <>PM2.5 <b>{nearestUserArea.region.pm25} µg/m³</b> · {getAirQuality(nearestUserArea.region.pm25).label}</>
                            : 'No fresh PM2.5 estimate available for this area'}
                      </p>
                    </>
                  ) : (
                    <p className="location-coordinates">
                      {userLocation.lat.toFixed(4)}° N, {userLocation.lon.toFixed(4)}° E
                    </p>
                  )}
                </div>
                <Check className="location-check" size={18} />
              </article>
            )}

            <article className="panel areas-panel">
              <div className="panel-heading areas-heading">
                <div>
                  <p className="section-kicker">EXPLORE THE CITY</p>
                  <h2>Nearby areas</h2>
                </div>
                <span className="areas-total">{listedAreas.length} areas</span>
              </div>
              <form className="area-search" onSubmit={handleFindClosest}>
                <label className="search-field">
                  <MapPin size={16} />
                  <input
                    type="search"
                    placeholder="Find a neighborhood"
                    value={search}
                    onChange={(event) => {
                      setSearch(event.target.value);
                      setNearestResult(null);
                      setNearestError('');
                    }}
                    aria-label="Find a neighborhood"
                  />
                </label>
                {search.trim().length >= 2 && matchingRegions.length === 0 && !nearestResult && (
                  <button className="closest-search-button" type="submit" disabled={lookingUpNearest}>
                    {lookingUpNearest ? 'Looking up area…' : 'Find closest listed area'}
                  </button>
                )}
              </form>
              {nearestError && <p className="nearest-error" role="alert">{nearestError}</p>}
              {nearestResult && (
                <p className="nearest-result-note" role="status">
                  Closest listed area to “{nearestResult.searched_area}” · {nearestResult.distance_km} km away
                </p>
              )}
              <div className="area-list">
                {filteredRegions.length === 0 ? (
                  <p className="no-results">
                    {regionsError
                      ? `Neighborhood names are available, but live pollution readings are offline. ${regionsError}`
                      : areasError || (regionsLoading ? 'Connecting to official monitoring stations…' : 'No neighborhoods match your search.')}
                  </p>
                ) : filteredRegions.map((region) => {
                  const quality = getRegionQuality(region, mapMetric);
                  const isPinned = watchlist.some((item) => item.name === region.name);
                  return (
                    <div className="area-row" key={region.id ?? region.name}>
                      <span className={`area-dot ${quality.tone}`} />
                      <div className="area-copy">
                        <strong>{region.name}</strong>
                        <span>
                          {mapMetric === 'aqi'
                            ? Number.isFinite(region.indicative_aqi)
                              ? <>
                                Indicative AQI <b>{region.indicative_aqi}</b> · {quality.label}
                                {region.indicative_aqi_dominant_pollutant
                                  ? ` · driven by ${region.indicative_aqi_dominant_pollutant}`
                                  : ''}
                                {' · '}{region.is_interpolated ? 'estimated' : 'measured'}
                                {getReadingAgeLabel(region.observed_at)}
                              </>
                              : 'Indicative AQI unavailable · needs PM2.5/PM10 and 3 current pollutants'
                            : Number.isFinite(region.pm25)
                              ? <>
                                PM2.5 <b>{region.pm25}</b> · {quality.label}
                                {' · '}{region.is_interpolated ? 'estimated' : 'measured'}
                                {getReadingAgeLabel(region.observed_at)}
                              </>
                              : regionsError
                                ? 'Live reading unavailable while station data is offline'
                                : 'No fresh official reading nearby'}
                        </span>
                      </div>
                      <button
                        className={`pin-button ${isPinned ? 'is-pinned' : ''}`}
                        onClick={() => addToWatchlist(region)}
                        disabled={isPinned}
                        aria-label={isPinned ? `${region.name} is in your watchlist` : `Pin ${region.name}`}
                      >
                        {isPinned ? <Check size={16} /> : <Plus size={16} />}
                      </button>
                    </div>
                  );
                })}
              </div>
            </article>
          </aside>

          <section className="panel map-panel">
            <div className="map-heading">
              <div>
                <p className="section-kicker">CITYWIDE AREA COVERAGE</p>
                <h2>Air quality map</h2>
                <p className="map-subtitle">
                  {mapMetric === 'aqi'
                    ? 'Indicative AQI uses recent pollutant readings; it is not the official time-averaged CPCB AQI.'
                    : 'PM2.5 concentration from CPCB/KSPCB stations and distance-weighted area estimates.'}
                </p>
              </div>
              <div className="map-controls">
                <div className="map-control-row">
                  <div className="map-mode-toggle" role="group" aria-label="Map metric">
                    <button
                      type="button"
                      className={mapMetric === 'pm25' ? 'is-selected' : ''}
                      aria-pressed={mapMetric === 'pm25'}
                      onClick={() => setMapMetric('pm25')}
                    >
                      PM2.5
                    </button>
                    <button
                      type="button"
                      className={mapMetric === 'aqi' ? 'is-selected' : ''}
                      aria-pressed={mapMetric === 'aqi'}
                      onClick={() => setMapMetric('aqi')}
                    >
                      Indicative AQI
                    </button>
                  </div>
                  <button className="map-action" onClick={handleGetLocation} disabled={loadingGeo}>
                    <Navigation size={15} /> Locate me
                  </button>
                </div>
              </div>
            </div>
            <div className="map-wrap">
              <MapContainer
                center={[12.9716, 77.5946]}
                zoom={11}
                scrollWheelZoom
                className="pollution-map"
              >
                <MapResizeHandler />
                <FitRegions regions={regions} userLocation={userLocation} />
                <FocusNearestRegion result={nearestResult} />
                <FocusUserLocation location={userLocation} />
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {mapRegions.map((region) => {
                  const value = getRegionMetric(region, mapMetric);
                  const quality = getRegionQuality(region, mapMetric);
                  const valueLabel = mapMetric === 'aqi'
                    ? `Indicative AQI ${Math.round(value)}`
                    : `PM2.5 ${value} µg/m³`;
                  return (
                    <Marker
                      key={region.id ?? region.name}
                      position={[region.lat, region.lon]}
                      icon={regionIcons.get(region.id ?? region.name)}
                      title={`${region.name}: ${valueLabel}, ${quality.label} (${region.is_interpolated ? 'estimated' : 'measured'})`}
                    >
                      <Popup>
                        <div className="map-popup">
                          <span className={`area-dot ${quality.tone}`} />
                          <div>
                            <strong>{region.name}</strong>
                            <p>
                              {mapMetric === 'aqi'
                                ? <>Indicative AQI <b>{Math.round(value)}</b> · {quality.label}</>
                                : <>PM2.5 <b>{value} µg/m³</b> · {quality.label}</>}
                              {' · '}{region.is_interpolated ? 'estimated' : 'measured'}
                            </p>
                            {mapMetric === 'aqi' && region.indicative_aqi_dominant_pollutant && (
                              <p>Highest sub-index: {region.indicative_aqi_dominant_pollutant}</p>
                            )}
                            {mapMetric === 'aqi' && (
                              <p className="aqi-caveat">{region.indicative_aqi_basis}</p>
                            )}
                            {region.source && <p>Source: {region.source}</p>}
                            {region.is_interpolated && region.source_station_count > 0 && (
                              <p>
                                Estimated from {region.source_station_count} nearby station(s)
                                {Number.isFinite(region.nearest_station_distance_km)
                                  ? ` · nearest ${region.nearest_station_distance_km} km`
                                  : ''}
                              </p>
                            )}
                            {region.observed_at && <p>Updated: {new Date(region.observed_at).toLocaleString()}</p>}
                            <button onClick={() => addToWatchlist(region)}>
                              <Plus size={13} /> Add to watchlist
                            </button>
                          </div>
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}
                {userLocation && (
                  <Marker position={[userLocation.lat, userLocation.lon]}>
                    <Popup>
                      <div className="map-popup">
                        <span className="area-dot good" />
                        <div>
                          <strong>Your current location</strong>
                          {nearestUserArea && (
                            <>
                              <p>
                                Nearest listed area: <b>{nearestUserArea.region.name}</b>
                                {' · '}{nearestUserArea.distance.toFixed(1)} km
                                {getReadingAgeLabel(nearestUserArea.region.observed_at)}
                              </p>
                              <p>
                                {regionsError
                                  ? 'Live pollution score unavailable while station data is offline'
                                  : mapMetric === 'aqi'
                                  ? Number.isFinite(nearestUserArea.region.indicative_aqi)
                                    ? <>Indicative AQI <b>{nearestUserArea.region.indicative_aqi}</b> · {getIndicativeAqiQuality(nearestUserArea.region.indicative_aqi).label}</>
                                    : 'Indicative AQI unavailable for this area'
                                  : Number.isFinite(nearestUserArea.region.pm25)
                                    ? <>PM2.5 <b>{nearestUserArea.region.pm25} µg/m³</b> · {getAirQuality(nearestUserArea.region.pm25).label}</>
                                    : 'No fresh PM2.5 estimate available for this area'}
                              </p>
                            </>
                          )}
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                )}
              </MapContainer>
              <div className="map-legend">
                <span><i className="tiny-dot good" /> Good</span>
                <span><i className="tiny-dot satisfactory" /> Satisfactory</span>
                <span><i className="tiny-dot moderate" /> Moderate</span>
                <span><i className="tiny-dot poor" /> Poor</span>
                <span><i className="tiny-dot very-poor" /> Very poor</span>
                <span><i className="tiny-dot severe" /> Severe</span>
              </div>
              {mapMetric === 'aqi' && mapRegions.length === 0 && (
                <div className="map-data-notice" role="status">
                  No locations currently have enough recent pollutant readings for an indicative AQI.
                </div>
              )}
            </div>
            <div className="map-footer">
              <span>
                <span className="status-dot" />
                {mapRegions.length} readings shown · {regions.length - mapRegions.length} unavailable
              </span>
              <span>{mapMetric === 'aqi' ? 'Indicative AQI · not official CPCB AQI' : 'PM2.5 · µg/m³'}</span>
            </div>
          </section>
        </section>

        <footer className="dashboard-footer">
          <span>Made for a more breathable Bengaluru.</span>
          <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">
            Map data © OpenStreetMap <ArrowUpRight size={13} />
          </a>
        </footer>
      </main>
    </div>
  );
}
