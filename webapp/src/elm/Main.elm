module Main exposing (..)

import Architecture exposing (Architecture, decodeArchitectures)
import Browser
import Html exposing (Html, a, div, h1, img, label, option, p, select, text)
import Http


type alias Model =
    { architectures : List Architecture
    , error : Maybe String
    }


type Msg
    = GotArchitectures (Result Http.Error (List Architecture))


init : () -> ( Model, Cmd Msg )
init _ =
    ( { architectures = [], error = Nothing }, fetchAvailableArchitectures )


fetchAvailableArchitectures : Cmd Msg
fetchAvailableArchitectures =
    Http.get
        { url = "/api/architectures"
        , expect = Http.expectJson GotArchitectures decodeArchitectures
        }


view : Model -> Html Msg
view model =
    case model.error of
        Nothing ->
            div []
                []

        Just msg ->
            div []
                [ p [] [ text msg ]
                ]


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        GotArchitectures r ->
            case r of
                Ok architectures ->
                    ( { model | architectures = architectures }, Cmd.none )

                Err e ->
                    ( { model | error = Just (errorToString e) }, Cmd.none )


errorToString : Http.Error -> String
errorToString e =
    case e of
        Http.BadUrl s ->
            "Bad url: " ++ s

        Http.Timeout ->
            "A network timeout occured"

        Http.NetworkError ->
            "A network error occurred"

        Http.BadStatus s ->
            "Bad status: " ++ String.fromInt s

        Http.BadBody s ->
            "Bad body: " ++ s


main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions =
            \_ ->
                Sub.none
        }
