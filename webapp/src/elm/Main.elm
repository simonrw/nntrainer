module Main exposing (..)

import Architecture exposing (Architecture, decodeArchitectures)
import Browser
import File exposing (File)
import File.Select as Select
import Html exposing (Html, a, button, div, h1, img, input, label, li, option, p, select, text, ul)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Http
import Json.Decode as D


type alias Model =
    { architectures : List Architecture
    , chosenArchitecture : Maybe Architecture
    , error : Maybe String
    }


type Msg
    = GotArchitectures (Result Http.Error (List Architecture))
    | ArchitectureChanged String
    | FileUploadRequested
    | FileSelected File
    | Uploaded (Result Http.Error ())


init : () -> ( Model, Cmd Msg )
init _ =
    ( { architectures = [], chosenArchitecture = Nothing, error = Nothing }, fetchAvailableArchitectures )


fetchAvailableArchitectures : Cmd Msg
fetchAvailableArchitectures =
    Http.get
        { url = "/api/architectures"
        , expect = Http.expectJson GotArchitectures decodeArchitectures
        }


onChange : (String -> msg) -> Html.Attribute msg
onChange handler =
    on "change" <| D.map handler <| D.at [ "target", "value" ] D.string


view : Model -> Html Msg
view model =
    case model.error of
        Nothing ->
            div []
                [ select [ onChange (\v -> ArchitectureChanged v) ]
                    (List.map
                        (\t -> option [] [ text <| Architecture.name t ])
                        model.architectures
                    )
                , button [ onClick FileUploadRequested ] [ text "Upload file" ]
                ]

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

        ArchitectureChanged a ->
            ( { model | chosenArchitecture = Just (Architecture.fromString a) }, Cmd.none )

        FileUploadRequested ->
            ( model, Select.file [] FileSelected )

        FileSelected f ->
            ( model
            , Http.request
                { method = "POST"
                , url = "/api/upload"
                , headers = []
                , body = Http.multipartBody [ Http.filePart "file" f ]
                , expect = Http.expectWhatever Uploaded
                , timeout = Nothing
                , tracker = Nothing
                }
            )

        Uploaded result ->
            case result of
                Ok _ ->
                    ( model, Cmd.none )

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
