module Main exposing (..)

import Browser
import Html exposing (Html, a, div, h1, img, label, option, p, select, text)


type alias Model =
    Int


type Msg
    = NoOp


init : () -> ( Model, Cmd Msg )
init _ =
    ( 0, Cmd.none )


view : Model -> Html Msg
view _ =
    div [] []


update : Msg -> Model -> ( Model, Cmd Msg )
update _ model =
    ( model, Cmd.none )


main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions =
            \_ ->
                Sub.none
        }
