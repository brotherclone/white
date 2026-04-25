declare namespace React {
  namespace JSX {
    interface IntrinsicElements {
      "midi-player": React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement> & {
          src?: string;
          "sound-font"?: string;
        },
        HTMLElement
      >;
    }
  }
}
