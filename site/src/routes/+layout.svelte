<script>
  let entries = [
    { name: "home", path: "/" },
    { name: "github", path: "https://github.com/acmiyaguchi/judo-footage-analysis"}
  ];
</script>

<svelte:head>
  <meta
    name="keywords"
    content="judo, multimedia, computer vision, research"
  />
  <meta name="author" content="Anthony Miyaguchi" />
  <meta name="title" property="og:title" content="Judo Footage Analysis" />
  <meta
    name="description"
    property="og:description"
    content="A project to analyze judo footage using computer vision techniques."
  />
</svelte:head>

<main>
  <nav>
    <div>
      {#each entries as entry}
        <a href={entry.path}>[{entry.name}]</a>
        <span />
      {/each}
    </div>
  </nav>
  <slot />
</main>

<style>
  main {
    margin-top: 1rem;
    padding-bottom: 3rem;
    margin: 0 auto;
    max-width: 992px;
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
  }

  @media (max-width: 992px) {
    main {
      padding: 0 0.5rem;
    }
  }
</style>
