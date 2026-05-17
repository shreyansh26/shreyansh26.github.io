# frozen_string_literal: true

# Ensure Net::HTTP uses an explicit CA bundle to avoid CRL lookup failures
# with OpenSSL 3.x on macOS when fetching Twitter oEmbed data.
require "net/http"
require "openssl"

module TwitterJekyll
  class ApiClient
    def fetch(api_request)
      uri = api_request.to_uri
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = api_request.ssl?
      http.read_timeout = 5
      http.open_timeout = 5

      store = OpenSSL::X509::Store.new
      store.add_file(ENV.fetch("SSL_CERT_FILE", "/etc/ssl/cert.pem"))
      http.cert_store = store

      response = http.start { |h| h.get uri.request_uri, REQUEST_HEADERS }

      handle_response(api_request, response)
    rescue Timeout::Error => e
      ErrorResponse.new(api_request, e.class.name).to_h
    end
  end
end
